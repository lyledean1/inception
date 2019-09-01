package image

import (
	"bytes"
	"github.com/lyledean1/inception/tensor"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"sort"
)

type Service interface {
	ProcessImageWithLabelPredictions(imageName string, imageBuffer bytes.Buffer) (*tensor.ClassifyResult, error)
	MakeTensorFromImage(imageBuffer *bytes.Buffer, imageFormat string) (*tf.Tensor, error)
	MakeTransformImageGraph(imageFormat string) (graph *tf.Graph, input, output tf.Output, err error)
}

type service struct {
	tensorService tensor.Service
}

func NewService(tensorService tensor.Service) Service {
	return &service{
		tensorService: tensorService,
	}
}

func (is *service) ProcessImageWithLabelPredictions(imageName string, imageBuffer bytes.Buffer) (*tensor.ClassifyResult, error) {
	sessionModel := is.tensorService.GetSessionModel()
	graphModel := is.tensorService.GetGraphModel()

	tensorFlow, err := is.MakeTensorFromImage(&imageBuffer, imageName)
	if err != nil {
		return nil, err
	}

	// Run inference
	output, err := sessionModel.Run(
		map[tf.Output]*tf.Tensor{
			graphModel.Operation("input").Output(0): tensorFlow,
		},
		[]tf.Output{
			graphModel.Operation("output").Output(0),
		},
		nil)
	if err != nil {
		return nil, err
	}

	classifyResult := tensor.ClassifyResult{
		Filename: imageName,
		Labels:   is.FindBestLabels(output[0].Value().([][]float32)[0]),
	}
	return &classifyResult, nil
}

func (is *service) MakeTensorFromImage(imageBuffer *bytes.Buffer, imageFormat string) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(imageBuffer.String())
	if err != nil {
		return nil, err
	}
	graph, input, output, err := is.MakeTransformImageGraph(imageFormat)
	if err != nil {
		return nil, err
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

// Creates a graph to decode, rezise and normalize an image
func (is *service) MakeTransformImageGraph(imageFormat string) (graph *tf.Graph, input, output tf.Output, err error) {
	const (
		H, W  = 224, 224
		Mean  = float32(117)
		Scale = float32(1)
	)
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	// Decode PNG or JPEG
	var decode tf.Output
	if imageFormat == "png" {
		decode = op.DecodePng(s, input, op.DecodePngChannels(3))
	} else {
		decode = op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
	}
	// Div and Sub perform (value-Mean)/Scale for each pixel
	output = op.Div(s,
		op.Sub(s,
			// Resize to 224x224 with bilinear interpolation
			op.ResizeBilinear(s,
				// Create a batch containing a single image
				op.ExpandDims(s,
					// Use decoded pixel values
					op.Cast(s, decode, tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func (is *service) FindBestLabels(probabilities []float32) []tensor.LabelResult {
	// Make a list of label/probability pairs
	var resultLabels []tensor.LabelResult
	labels := is.tensorService.GetLabels()
	for i, p := range probabilities {
		if i >= len(labels) {
			break
		}
		resultLabels = append(resultLabels, tensor.LabelResult{Label: labels[i], Probability: p})
	}
	// Sort by probability
	sort.Sort(tensor.ByProbability(resultLabels))
	// Return top 5 labels
	return resultLabels[0:10]
}
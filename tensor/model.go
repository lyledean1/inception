package tensor

import (
	"bufio"
	"io/ioutil"
	"log"
	"os"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)


var (
	graphModel   *tf.Graph
	sessionModel *tf.Session
	labels       []string
)

type Service interface {
	GetGraphModel() *tf.Graph
	GetSessionModel() *tf.Session
	GetLabels() []string
}

type service struct {
}

func NewService() Service {
	return &service{}
}

func (s *service) GetGraphModel() *tf.Graph {
	if graphModel == nil {
		LoadModel()
	}
	return graphModel
}

func (s *service) GetSessionModel() *tf.Session {
	if sessionModel == nil {
		LoadModel()
	}
	return sessionModel
}

func (s *service) GetLabels() []string {
	if len(labels) == 0 {
		LoadModel()
	}
	return labels
}

func LoadModel() error {
	// Load inception model
	model, err := ioutil.ReadFile("/model/tensorflow_inception_graph.pb")
	if err != nil {
		return err
	}
	graphModel = tf.NewGraph()
	if err := graphModel.Import(model, ""); err != nil {
		return err
	}

	sessionModel, err = tf.NewSession(graphModel, nil)
	if err != nil {
		log.Fatal(err)
	}

	// Load labels
	labelsFile, err := os.Open("/model/imagenet_comp_graph_label_strings.txt")
	if err != nil {
		return err
	}
	defer labelsFile.Close()
	scanner := bufio.NewScanner(labelsFile)
	// Labels are separated by newlines
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

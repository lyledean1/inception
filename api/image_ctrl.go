package api

import (
	"bytes"
	"io"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/lyledean1/inception/image"
)

type ImageController struct {
	service image.Service
}

func NewImageController(service image.Service) *ImageController {
	return &ImageController{
		service: service,
	}
}

func (ctrl *ImageController) PredictLabelFromImage(c *gin.Context) {
	imageFile, header, err := c.Request.FormFile("image")
	if err != nil {
		c.JSON(400, nil)
		return
	}
	imageName := strings.Split(header.Filename, ".")[:1][0]
	defer imageFile.Close()
	var imageBuffer bytes.Buffer
	_, err = io.Copy(&imageBuffer, imageFile)
	if err != nil {
		c.JSON(400, nil)
		return
	}
	cmd, err := ctrl.service.ProcessImageWithLabelPredictions(imageName, imageBuffer)
	if err != nil {
		c.JSON(400, nil)
		return
	}
	c.JSON(200, cmd)
}

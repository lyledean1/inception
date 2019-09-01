package api

import (
	"github.com/gin-gonic/gin"
	"github.com/lyledean1/inception/image"
)


func SetupRouter(imageService image.Service) *gin.Engine {
	gin.SetMode(gin.DebugMode)
	r := gin.New()
	r.RedirectTrailingSlash = false
	r.Use(gin.Logger())
	r.Use(gin.Recovery())
	imageCtrl := NewImageController(imageService)
	image := r.Group("/api/v1/image")
	image.POST("", imageCtrl.PredictLabelFromImage)
	r.GET("/", func(c *gin.Context) {
		//health check
		c.JSON(200, nil)
	})
	return r
}

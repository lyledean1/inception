package api

import (
	"os"

	"github.com/gin-gonic/gin"
	"github.com/lyledean1/inception/image"
)

func SetupRouter(imageService image.Service) *gin.Engine {
	gin.SetMode(gin.DebugMode)
	r := gin.New()
	r.RedirectTrailingSlash = false
	r.Use(gin.Logger())
	r.Use(gin.Recovery())
	r.GET("/", func(c *gin.Context) {
		//health check
		c.JSON(200, nil)
	})
	imageCtrl := NewImageController(imageService)
	image := r.Group("/api/v1/image", gin.BasicAuth(gin.Accounts{
		"lyle": os.Getenv("INCEPTION_PASSWORD")}))
	image.POST("", imageCtrl.PredictLabelFromImage)
	return r
}

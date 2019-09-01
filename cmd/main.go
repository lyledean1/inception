package main

import (
	"github.com/lyledean1/inception/api"
	"github.com/lyledean1/inception/image"
	"github.com/lyledean1/inception/tensor"
)

func main() {
	if err := tensor.LoadModel(); err != nil {
		panic(err)
		return
	}
	tensorService := tensor.NewService()
	imageService := image.NewService(tensorService)
	r := api.SetupRouter(imageService)
	r.Run()
}
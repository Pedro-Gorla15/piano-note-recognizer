package main

// Cambio de comentario

import (
	"fmt"
	"path/filepath"
	"piano-note-recognizer/pkg/audio"
	"piano-note-recognizer/pkg/neuralnet"
	"strings"

	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
)

func prepareData() ([]*mat.Dense, []*mat.Dense, []string, []string) {
	notes := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	filesDir := "Notas_WAV"
	var trainData, validData []*mat.Dense
	var trainLabels, validLabels []string

	for i, note := range notes {
		filename := filepath.Join(filesDir, note+".wav")
		// Asegurarse que cada archivo de audio tenga al menos 1024 muestras
		audioData, err := audio.LoadAudio(filename, 1024)
		if err != nil {
			fmt.Printf("Error loading audio from %s: %v\n", filename, err)
			continue
		}
		spectrogram := audio.GenerateSpectrogram(audioData, 1024)
		normalizedSpectrogram := audio.NormalizeSpectrogram(spectrogram)

		if audio.CheckNaN(normalizedSpectrogram) {
			fmt.Printf("NaN detected in normalized spectrogram for note: %s, skipping...\n", note)
			continue
		}

		for _, slice := range normalizedSpectrogram {
			if len(slice) != 1024 {
				fmt.Printf("Invalid slice length for note: %s, expected 1024, got %d\n", note, len(slice))
				continue
			}
			matrix := mat.NewDense(1, 1024, slice) // cambio
			if i%5 == 0 {
				validData = append(validData, matrix)
				validLabels = append(validLabels, note)
			} else {
				trainData = append(trainData, matrix)
				trainLabels = append(trainLabels, note)
			}
		}
	}

	return trainData, validData, trainLabels, validLabels
}

func TestModel(rnn *neuralnet.RNN, filename string) {
	audioData, err := audio.LoadAudio(filename, 1024)
	if err != nil {
		fmt.Printf("Error loading audio from %s: %v\n", filename, err)
		return
	}
	spectrogram := audio.GenerateSpectrogram(audioData, 1024)
	normalizedSpectrogram := audio.NormalizeSpectrogram(spectrogram)

	if audio.CheckNaN(normalizedSpectrogram) {
		fmt.Println("NaN detected in normalized spectrogram, cannot proceed with predictions.")
		return
	}

	var predictions []string
	for i, segment := range normalizedSpectrogram {
		if len(segment) != 1024 {
			fmt.Printf("Skipping segment %d due to incorrect length: %d\n", i, len(segment))
			continue
		}
		segmentMatrix := mat.NewDense(1024, 1, segment)
		segmentTensor := tensor.New(tensor.WithShape(1, 1024), tensor.Of(tensor.Float64), tensor.WithBacking(segmentMatrix.RawMatrix().Data))

		predictedNotes := rnn.Predict(segmentTensor)
		if len(predictedNotes) == 0 {
			fmt.Printf("No predictions for segment %d\n", i)
		}
		predictions = append(predictions, predictedNotes...)
	}

	if len(predictions) == 0 {
		fmt.Println("No notes were predicted.")
	} else {
		fmt.Printf("Notes in file: %s\n", strings.Join(predictions, ", "))
	}
}

func main() {
	trainData, validData, trainLabels, validLabels := prepareData()
	if len(trainData) == 0 || len(validData) == 0 {
		fmt.Println("Training or validation data is empty, cannot proceed.")
		return
	}
	fmt.Println("Data and labels prepared for training and validation.")

	config := neuralnet.NetworkConfig{
		InputSize:  1024,
		HiddenSize: 128,
		OutputSize: 12,
	}
	rnn := neuralnet.NewRNN(config)
	fmt.Println("Starting training...")
	rnn.Train(trainData, trainLabels, 10)
	fmt.Println("Training completed.")

	fmt.Println("Validating model...")
	neuralnet.Validate(rnn, validData, validLabels)

	testFilename := filepath.Join("Notas_WAV", "A.wav")
	fmt.Println("Testing with file:", testFilename)
	TestModel(rnn, testFilename)
}

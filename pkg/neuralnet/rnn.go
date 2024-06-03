package neuralnet

import (
	"fmt"
	"log"
	"math"

	"gonum.org/v1/gonum/mat"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type NetworkConfig struct {
	InputSize  int
	HiddenSize int
	OutputSize int
}

type RNN struct {
	g      *gorgonia.ExprGraph
	w, h0  *gorgonia.Node // Peso y estado oculto inicial
	outW   *gorgonia.Node // Peso de la capa de salida
	vm     gorgonia.VM    // Máquina virtual para ejecutar el grafo
	Config NetworkConfig  // Configuración de la red
}

func NewRNN(config NetworkConfig) *RNN {
	g := gorgonia.NewGraph()
	inputSize, hiddenSize, outputSize := config.InputSize, config.HiddenSize, config.OutputSize

	fmt.Printf("Initializing RNN with inputSize: %d, hiddenSize: %d, outputSize: %d\n", inputSize, hiddenSize, outputSize)

	w := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(inputSize+hiddenSize, hiddenSize), gorgonia.WithName("w"), gorgonia.WithInit(gorgonia.GlorotU(1)))
	h0 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, hiddenSize), gorgonia.WithName("h0"), gorgonia.WithInit(gorgonia.Zeroes()))
	outW := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(hiddenSize, outputSize), gorgonia.WithName("outW"), gorgonia.WithInit(gorgonia.GlorotU(1)))

	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(w, h0, outW))

	return &RNN{
		g:      g,
		w:      w,
		h0:     h0,
		outW:   outW,
		vm:     vm,
		Config: config,
	}
}

func (r *RNN) forward(x *tensor.Dense) (*gorgonia.Node, error) {
	if x.Shape()[0] != 1 || x.Shape()[1] != r.Config.InputSize {
		return nil, fmt.Errorf("input tensor has incorrect shape, expected [1, %d], got %v", r.Config.InputSize, x.Shape())
	}

	xNode := gorgonia.NewTensor(r.g, tensor.Float64, 2, gorgonia.WithShape(x.Shape()...), gorgonia.WithValue(x))
	if xNode.Value() == nil {
		return nil, fmt.Errorf("xNode value is nil")
	}

	if r.h0 == nil {
		r.h0 = gorgonia.NewMatrix(r.g, tensor.Float64, gorgonia.WithShape(1, r.Config.HiddenSize), gorgonia.WithInit(gorgonia.Zeroes()))
	}

	concatenated, err := gorgonia.Concat(1, xNode, r.h0)
	if err != nil {
		return nil, fmt.Errorf("failed to concatenate input and hidden state: %v", err)
	}

	mulResult, err := gorgonia.Mul(concatenated, r.w)
	if err != nil {
		return nil, fmt.Errorf("failed matrix multiplication: %v", err)
	}

	r.h0 = gorgonia.Must(gorgonia.Tanh(mulResult)) // Agrega activación

	output, err := gorgonia.Mul(r.h0, r.outW)
	if err != nil {
		return nil, fmt.Errorf("failed to add output weights: %v", err)
	}

	if outputHasNaNOrInf(output) {
		return nil, fmt.Errorf("output has NaN or Inf values")
	}

	return output, nil
}

func outputHasNaNOrInf(output *gorgonia.Node) bool {
	if output.Value() == nil {
		fmt.Println("Output value is nil")
		return true
	}
	data, ok := output.Value().Data().([]float64)
	if !ok {
		fmt.Println("Output data type is not []float64")
		return true
	}
	hasNaN := false
	hasInf := false
	for i, v := range data {
		if math.IsNaN(v) {
			fmt.Printf("NaN detected at index %d\n", i)
			hasNaN = true
		}
		if math.IsInf(v, 0) {
			fmt.Printf("Inf detected at index %d\n", i)
			hasInf = true
		}
	}
	return hasNaN || hasInf
}

func lossHasNaN(loss *gorgonia.Node) bool {
	lossValue := loss.Value().Data().(float64)
	return math.IsNaN(lossValue)
}

func (r *RNN) Train(data []*mat.Dense, labels []string, epochs int) {
	fmt.Println("Starting training...")
	var tensorData []*tensor.Dense
	for _, d := range data {
		rawData := d.RawMatrix().Data
		t := tensor.New(tensor.WithShape(d.Dims()), tensor.Of(tensor.Float64), tensor.WithBacking(rawData))
		tensorData = append(tensorData, t)
	}

	labelTensors, err := prepareLabels(labels, r.Config.OutputSize, r.g)
	if err != nil {
		log.Println("Error preparing labels:", err)
		return
	}

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i, t := range tensorData {
			reshapedTensor := tensor.New(tensor.WithShape(1, 1024), tensor.Of(tensor.Float64), tensor.WithBacking(t.Data()))
			output, err := r.forward(reshapedTensor)
			if err != nil {
				log.Println("Error in forward pass:", err)
				continue
			}

			if outputHasNaNOrInf(output) {
				fmt.Printf("NaN or Inf detected in output at epoch %d, index %d\n", epoch, i)
				continue
			}

			diff := gorgonia.Must(gorgonia.Sub(output, labelTensors[i]))
			sqDiff := gorgonia.Must(gorgonia.Square(diff))
			loss := gorgonia.Must(gorgonia.Mean(sqDiff))

			if lossHasNaN(loss) {
				fmt.Printf("NaN detected in loss at epoch %d, index %d\n", epoch, i)
				continue
			}

			if _, err := gorgonia.Grad(loss, r.w, r.h0, r.outW); err != nil {
				log.Println("Error computing gradients:", err)
				continue
			}

			r.clipGradients(5.0)

			if err = r.vm.RunAll(); err != nil {
				log.Println("Error running the computation graph:", err)
				continue
			}
			r.vm.Reset()

			totalLoss += loss.Value().Data().(float64)
		}
		averageLoss := totalLoss / float64(len(tensorData))
		fmt.Printf("Epoch %d: Average Loss: %.4f\n", epoch, averageLoss)
	}
}

func (r *RNN) clipGradients(maxNorm float64) {
	for _, node := range []*gorgonia.Node{r.w, r.h0, r.outW} {
		gradVal, _ := node.Grad()
		gradTensor, _ := gradVal.(*tensor.Dense)
		data := gradTensor.Data().([]float64)
		norm := 0.0
		for _, v := range data {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if norm > maxNorm {
			scale := maxNorm / norm
			for i := range data {
				data[i] *= scale
			}
		}
	}
}

func prepareLabels(labels []string, numClasses int, g *gorgonia.ExprGraph) ([]*gorgonia.Node, error) {
	if numClasses <= 0 {
		return nil, fmt.Errorf("invalid number of classes: %d", numClasses)
	}

	var result []*gorgonia.Node
	for _, label := range labels {
		index, err := labelToIndex(label)
		if err != nil {
			return nil, err // Propagar el error hacia arriba
		}
		if index < 0 || index >= numClasses {
			return nil, fmt.Errorf("index out of range [%d] with numClasses %d", index, numClasses)
		}

		oneHot := make([]float64, numClasses)
		oneHot[index] = 1.0
		t := tensor.New(tensor.WithBacking(oneHot), tensor.WithShape(numClasses))
		node := gorgonia.NewTensor(g, tensor.Float64, 1, gorgonia.WithValue(t))
		result = append(result, node)
	}
	return result, nil
}

// labelToIndex convierte una etiqueta de nota musical a un índice numérico basado en el orden de las notas
func labelToIndex(label string) (int, error) {
	// Mapa de notas a índices
	noteToIndex := map[string]int{
		"C": 0, "C#": 1, "D": 2, "D#": 3,
		"E": 4, "F": 5, "F#": 6, "G": 7,
		"G#": 8, "A": 9, "A#": 10, "B": 11,
	}
	index, exists := noteToIndex[label]
	if !exists {
		return -1, fmt.Errorf("label '%s' is not a valid note", label)
	}
	return index, nil
}

func Validate(rnn *RNN, validData []*mat.Dense, validLabels []string) {
	var totalLoss float64
	for i, vData := range validData {
		vTensor := tensor.New(tensor.WithShape(vData.Dims()), tensor.Of(tensor.Float64), tensor.WithBacking(vData.RawMatrix().Data))
		output, err := rnn.forward(vTensor)
		if err != nil {
			log.Printf("Error during validation forward pass: %v\n", err)
			continue
		}

		labelTensor, err := prepareLabels([]string{validLabels[i]}, rnn.Config.OutputSize, rnn.g)
		if err != nil {
			log.Printf("Error preparing labels: %v\n", err)
			continue
		}

		diff := gorgonia.Must(gorgonia.Sub(output, labelTensor[0]))
		sqDiff := gorgonia.Must(gorgonia.Square(diff))
		loss := gorgonia.Must(gorgonia.Mean(sqDiff))

		totalLoss += loss.Value().Data().(float64)
	}
	if len(validData) > 0 {
		averageLoss := totalLoss / float64(len(validData))
		fmt.Printf("Validation Loss: %.4f\n", averageLoss)
	} else {
		fmt.Println("No valid data provided for validation.")
	}
}

func (r *RNN) Predict(x tensor.Tensor) []string {
	fmt.Println("Predicting...")
	if r.vm == nil {
		fmt.Println("No VM initialized")
		return nil
	}

	if _, ok := x.Data().([]float64); !ok {
		fmt.Println("Invalid tensor data type; expected []float64")
		return nil
	}
	if x.Shape()[0] != 1 || x.Shape()[1] != r.Config.InputSize {
		fmt.Printf("Invalid input shape; expected [1, %d], got %v\n", r.Config.InputSize, x.Shape())
		return nil
	}

	denseX, ok := x.(*tensor.Dense)
	if !ok {
		fmt.Println("Tensor is not of type *tensor.Dense as required")
		return nil
	}

	output, err := r.forward(denseX)
	if err != nil {
		fmt.Printf("Error during prediction: %v\n", err)
		return nil
	}

	outputData, ok := output.Value().Data().([]float64)
	if !ok {
		fmt.Println("Output data type assertion failed")
		return nil
	}

	probs := softmax(outputData)
	threshold := 0.5
	predictedNotes := decodePredictions(probs, threshold)
	return predictedNotes
}

func softmax(x []float64) []float64 {
	if len(x) == 0 {
		return nil
	}
	exps := make([]float64, len(x))
	maxVal := max(x) // Encuentra el valor máximo para evitar desbordamientos
	var sum float64

	for i, v := range x {
		exps[i] = math.Exp(v - maxVal) // Resta el valor máximo para la estabilidad numérica
		sum += exps[i]
	}

	for i := range exps {
		exps[i] /= sum // Divide cada exponente por la suma de todos los exponentes
	}
	return exps
}

// Función que ayuda a encontrar el máximo valor en un slice de tipo float64
func max(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	maxVal := values[0]
	for _, v := range values[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

// Decodificar las probabilidades a notas
func decodePredictions(probs []float64, threshold float64) []string {
	notes := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	var results []string
	numClasses := len(notes)
	if len(probs) < numClasses {
		numClasses = len(probs)
	}
	for i := 0; i < numClasses; i++ {
		if probs[i] > threshold {
			results = append(results, notes[i])
		}
	}
	return results
}

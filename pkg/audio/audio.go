package audio

import (
	"fmt"
	"math"
	"os"

	"github.com/go-audio/wav"
	"gonum.org/v1/gonum/dsp/fourier"
)

// LoadAudio carga un archivo de audio y retorna una slice de int con las muestras de audio
func LoadAudio(filename string, frameSize int) ([]int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error opening file %s: %w", filename, err)
	}
	defer file.Close()

	decoder := wav.NewDecoder(file)
	if !decoder.IsValidFile() {
		return nil, fmt.Errorf("invalid WAV file %s", filename)
	}

	buff, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, fmt.Errorf("error decoding WAV file %s: %w", filename, err)
	}
	if buff.NumFrames() == 0 {
		return nil, fmt.Errorf("empty audio buffer in WAV file %s", filename)
	}

	// Revisa si el número de muestras es menor que frameSize y aplica padding si es necesario
	numSamples := buff.NumFrames()
	if numSamples < frameSize {
		extendedData := make([]int, frameSize)
		copy(extendedData, buff.Data)
		// Rellenar el resto con ceros
		for i := numSamples; i < frameSize; i++ {
			extendedData[i] = 0
		}
		fmt.Printf("Audio from %s was padded from %d to %d samples\n", filename, numSamples, frameSize)
		return extendedData, nil
	}

	fmt.Printf("Loaded %d samples from %s\n", numSamples, filename)
	return buff.Data, nil
}

// GenerateSpectrogram genera un espectrograma a partir de datos de audio
func GenerateSpectrogram(data []int, frameSize int) [][]float64 {
	if frameSize <= 0 || frameSize > len(data) {
		return nil // Retorna nil para frameSize no válido
	}

	fft := fourier.NewFFT(frameSize)
	window := hannWindow(frameSize)
	spectrogram := make([][]float64, 0)

	for start := 0; start < len(data); start += frameSize {
		end := start + frameSize
		if end > len(data) {
			end = len(data)
			tempData := make([]int, frameSize) // Asegura que el buffer tenga siempre frameSize elementos
			copy(tempData, data[start:end])
			data = tempData
		} else {
			data = data[start:end]
		}

		frame := make([]float64, frameSize)
		for i := range frame {
			frame[i] = float64(data[i]) * window[i]
		}
		coeff := fft.Coefficients(nil, frame)
		power := make([]float64, frameSize) // Asegura que cada segmento tenga la longitud deseada
		for i := range power {
			if i < len(coeff)/2 {
				realPart := real(coeff[i])
				imagPart := imag(coeff[i])
				power[i] = realPart*realPart + imagPart*imagPart
			} else {
				power[i] = 0 // Rellena el resto con ceros
			}
			if math.IsNaN(power[i]) || math.IsInf(power[i], 0) {
				power[i] = 0
			}
		}
		spectrogram = append(spectrogram, power)
	}
	return spectrogram
}

func hannWindow(size int) []float64 {
	if size <= 1 { // Controla casos degenerados
		return []float64{1} // Una ventana de tamaño 1 no tiene efecto
	}

	window := make([]float64, size)
	piFactor := 2 * math.Pi / float64(size-1) // Factor pre-computado para reducir operaciones en el loop

	for i := range window {
		window[i] = 0.5 * (1 - math.Cos(piFactor*float64(i))) // Se aplica la formula de la ventana de Hann
	}
	return window
}

func NormalizeSpectrogram(spectrogram [][]float64) [][]float64 {
	for i, row := range spectrogram {
		maxVal := findMaxAbs(row)
		if maxVal == 0 { // Evitar división por cero
			continue
		}
		for j := range row {
			spectrogram[i][j] /= maxVal
			if math.IsNaN(spectrogram[i][j]) || math.IsInf(spectrogram[i][j], 0) {
				spectrogram[i][j] = 0 // Rellenar con ceros cualquier valor NaN o Inf
			}
		}
	}
	return spectrogram
}

// Helper function to find the maximum absolute value in a slice
func findMaxAbs(slice []float64) float64 {
	maxVal := 0.0
	for _, v := range slice {
		absV := math.Abs(v)
		if absV > maxVal {
			maxVal = absV
		}
	}
	return maxVal
}

func CheckNaN(data [][]float64) bool {
	for i, series := range data {
		for j, value := range series {
			if math.IsNaN(value) {
				fmt.Printf("NaN found at position [%d][%d]\n", i, j) // Opcional: Log de la posición
				return true
			}
		}
	}
	return false
}

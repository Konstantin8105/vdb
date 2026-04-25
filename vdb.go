package vdb

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/gob"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"
)

type Embeder struct {
	Model string // AI model name, e.g., "text-embedding-nomic-embed-text-v1.5"

	Endpoint string // API endpoint URL, e.g., "http://127.0.0.1:1234/v1"
	Key      string // API key for external providers (optional)

	ContextSize int // Maximum context window size in tokens

	RequestTimeout time.Duration
}

func (o Embeder) Calculate(text string) (code []float32, err error) {
	// Validate endpoint
	if o.Endpoint == "" {
		err = fmt.Errorf("empty endpoint")
		return
	}
	// Validate model name
	if o.Model == "" {
		err = fmt.Errorf("empty model name")
		return
	}
	// Set default timeout if not specified
	if o.RequestTimeout == 0 {
		o.RequestTimeout = 4 * time.Hour
	}

	endpoint := o.Endpoint
	// Ensure endpoint ends with slash for path concatenation
	if len(endpoint) > 0 && endpoint[len(endpoint)-1] != '/' {
		endpoint += "/"
	}
	endpoint += "embeddings"

	// log.Printf("AI-comp endpoint: %s", endpoint)

	pr := map[string]string{
		"input": text,
		"model": o.Model,
	}

	jsonData, err := json.Marshal(pr)
	if err != nil {
		err = fmt.Errorf("marshal error: %w", err)
		return
	}

	client := &http.Client{Timeout: o.RequestTimeout}
	req, err := http.NewRequest("POST", endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		err = fmt.Errorf("create request error: %w", err)
		return
	}
	// Set headers
	req.Header.Set("Content-Type", "application/json")
	if o.Key != "" {
		// For OpenAI-compatible APIs, Bearer token is typically used
		req.Header.Set("Authorization", "Bearer "+o.Key)
	}
	resp, err := client.Do(req)
	if err != nil {
		err = fmt.Errorf("http error: %w", err)
		return
	}
	// log.Printf("RouterAI response: %v", resp)
	defer func() {
		errC := resp.Body.Close()
		if errC != nil {
			if err != nil {
				err = errors.Join(err, errC)
			} else {
				err = errC
			}
		}
	}()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		err = fmt.Errorf("read error: %w", err)
		return
	}
	// log.Printf("RouterAI response data: %s", string(data))
	if resp.StatusCode != http.StatusOK {
		err = fmt.Errorf("status %d: %s", resp.StatusCode, string(data))
		return
	}

	rb := struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}{}
	if err = json.Unmarshal(data, &rb); err != nil {
		err = fmt.Errorf("unmarshal error: %w", err)
		return
	}
	if len(rb.Data) == 0 {
		err = fmt.Errorf("responce data is empty")
		return
	}
	if len(rb.Data[0].Embedding) == 0 {
		err = fmt.Errorf("reasponce data embedding is empty")
		return
	}

	// do not normalize
	code = rb.Data[0].Embedding
	return
}

type Document struct {
	ID      string
	Content string
	Code    []float32 // embedding code array

	filename string
}

func (doc Document) String() string {
	return fmt.Sprintf("Id: %s\n%s\n", doc.ID, doc.Content)
}

func (doc Document) hash() string {
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s:%s", doc.ID, doc.Content)))
	return hex.EncodeToString(hash[:])
}

func (doc *Document) Write(path string, compress bool) (err error) {
	if doc.filename == "" {
		doc.filename = filepath.Join(path, doc.hash()[:12]+getExt(compress))
	}
	var buf bytes.Buffer
	if compress {
		gz := gzip.NewWriter(&buf)
		enc := gob.NewEncoder(gz)
		err = enc.Encode(&doc)
		if err != nil {
			return
		}
		if err = gz.Close(); err != nil {
			return
		}
	} else {
		err = gob.NewEncoder(&buf).Encode(&doc)
		if err != nil {
			return
		}
	}
	err = os.WriteFile(doc.filename, buf.Bytes(), 0777)
	return
}

type Collection struct {
	Documents []*Document
	Embed     Embeder

	path     string
	compress bool
}

func getExt(compress bool) string {
	ext := ".gob"
	if compress {
		ext += ".gz"
	}
	return ext
}

func New(path string, compress bool, embed Embeder) (
	collection *Collection,
	err error,
) {
	// chack path
	if path == "" {
		err = fmt.Errorf("path is empty")
		return
	}
	path = filepath.Clean(path) // clean path
	_ = os.MkdirAll(path, os.ModePerm)
	// prepare empty
	collection = &Collection{
		Documents: make([]*Document, 0),
		Embed:     embed,
		path:      path,
		compress:  compress,
	}
	// if path is exist, then get
	var files []string
	files, err = filepath.Glob(filepath.Join(path, "*"+getExt(collection.compress)))
	if err != nil {
		return
	}
	if 0 < len(files) {
		var docs []*Document
		for _, file := range files {
			// read file and uncompress if need
			var dat []byte
			if collection.compress {
				var f *os.File
				f, err = os.Open(file)
				if err != nil {
					return
				}
				var gz *gzip.Reader
				gz, err = gzip.NewReader(f)
				if err != nil {
					return
				}
				dat, err = io.ReadAll(gz)
				if err != nil {
					return
				}
				err = gz.Close()
				if err != nil {
					return
				}
				err = f.Close()
				if err != nil {
					return
				}
			} else {
				dat, err = os.ReadFile(file)
				if err != nil {
					return
				}
			}
			var buf bytes.Buffer
			dec := gob.NewDecoder(&buf)
			_, err = buf.Write(dat)
			if err != nil {
				return
			}
			// unmarshaling
			var doc Document
			err = dec.Decode(&doc)
			if err != nil {
				return
			}
			doc.filename = file
			docs = append(docs, &doc)
		}
		err = collection.AddDocuments(docs...)
		if err != nil {
			return
		}
	}
	return
}

func (collection *Collection) AddDocuments(docs ...*Document) (err error) {
	if len(docs) == 0 {
		// empty docs
		return
	}
	// error handling
	for _, doc := range docs {
		if doc == nil {
			err = fmt.Errorf("doc is nil")
			return
		}
		if doc.ID == "" {
			err = fmt.Errorf("ID doc is empty")
			return
		}
		if doc.Content == "" {
			err = fmt.Errorf("content doc is empty")
		}
	}
	collection.Documents = append(collection.Documents, docs...)
	for i := range docs {
		if 0 < len(docs[i].Code) {
			continue
		}
		// update code
		var code []float32
		code, err = collection.Embed.Calculate(docs[i].Content)
		if err != nil {
			return
		}
		docs[i].Code = code
		err = docs[i].Write(collection.path, collection.compress)
		if err != nil {
			return
		}
	}
	return
}

func (collection *Collection) RemoveDocuments(filter func(doc *Document) (remove bool)) (err error) {
	if filter == nil {
		err = fmt.Errorf("filter is nil")
		return
	}
	collection.Documents = slices.DeleteFunc(collection.Documents, func(doc *Document) bool {
		remove := filter(doc)
		if remove {
			err1 := os.Remove(doc.filename)
			if err1 != nil {
				err = errors.Join(err, err1)
			}
		}
		return remove
	})
	return
}

type CompareVector uint8

const (
	CosineSimilarity CompareVector = iota
	EuclideanDistance
	ManhattanDistance
	DotProduct
	PearsonCorrelation
	ChebyshevDistance
)

// Calculate computes the selected metric between two vectors a and b.
// Assumes a and b have the same length. If lengths differ or length is zero,
// the function returns 0 (can also panic or return an error if desired).
// All metrics follow the rule: higher return value = more similar vectors.
func (c CompareVector) Calculate(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	n := len(a)

	switch c {
	case CosineSimilarity:
		// Cosine Similarity:
		// Formula: cos(θ) = (A·B) / (||A|| ||B||) = (∑ a_i b_i) / (√∑a_i² * √∑b_i²)
		// Description: measures the cosine of the angle between vectors.
		// Result range: [-1, 1]. Larger = more similar (1 = identical direction, -1 = opposite).
		// For non-negative embeddings, result lies in [0,1].
		var dot, normA, normB float64
		for i := range n {
			ai, bi := float64(a[i]), float64(b[i])
			dot += ai * bi
			normA += ai * ai
			normB += bi * bi
		}
		if normA == 0 || normB == 0 {
			return 0
		}
		return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))

	case EuclideanDistance:
		// Euclidean Distance converted to similarity:
		// Raw distance: d = √(∑ (a_i - b_i)²)
		// Converted similarity: sim = 1 / (1 + d)
		// Description: geometric distance between points, then inverted to follow "higher = more similar".
		// Result range: (0, 1]. 1 = identical vectors, approaches 0 as distance grows.
		var sum float64
		for i := range n {
			diff := float64(a[i] - b[i])
			sum += diff * diff
		}
		d := math.Sqrt(sum)
		return float32(1.0 / (1.0 + d))

	case ManhattanDistance:
		// Manhattan Distance converted to similarity:
		// Raw distance: d = ∑ |a_i - b_i|
		// Converted similarity: sim = 1 / (1 + d)
		// Description: sum of absolute differences, then inverted.
		// Result range: (0, 1]. 1 = identical vectors, approaches 0 as distance grows.
		var sum float64
		for i := range n {
			sum += math.Abs(float64(a[i] - b[i]))
		}
		return float32(1.0 / (1.0 + sum))

	case DotProduct:
		// Dot Product:
		// Formula: A·B = ∑ a_i b_i
		// Description: sum of products of corresponding components.
		// Result range: (-∞, +∞). Larger positive = more similar (assuming non-negative vectors).
		// For normalized (L2‑norm=1) vectors it equals cosine similarity.
		var dot float64
		for i := range n {
			dot += float64(a[i]) * float64(b[i])
		}
		return float32(dot)

	case PearsonCorrelation:
		// Pearson Correlation:
		// Formula: r = (∑ (a_i - μ_A)(b_i - μ_B)) / (√∑(a_i-μ_A)² * √∑(b_i-μ_B)²)
		// where μ_A = (1/n)∑a_i, μ_B = (1/n)∑b_i
		// Description: measures linear dependence; cosine similarity of centered vectors.
		// Result range: [-1, 1]. Larger = more similar (1 = perfect positive correlation).
		var meanA, meanB float64
		for i := range n {
			meanA += float64(a[i])
			meanB += float64(b[i])
		}
		meanA /= float64(n)
		meanB /= float64(n)

		var num, denA, denB float64
		for i := range n {
			ai := float64(a[i]) - meanA
			bi := float64(b[i]) - meanB
			num += ai * bi
			denA += ai * ai
			denB += bi * bi
		}
		if denA == 0 || denB == 0 {
			return 0
		}
		return float32(num / (math.Sqrt(denA) * math.Sqrt(denB)))

	case ChebyshevDistance:
		// Chebyshev Distance converted to similarity:
		// Raw distance: d = max_i |a_i - b_i|
		// Converted similarity: sim = 1 / (1 + d)
		// Description: maximum absolute difference, then inverted.
		// Result range: (0, 1]. 1 = identical vectors, approaches 0 as distance grows.
		var maxDiff float64
		for i := range n {
			diff := math.Abs(float64(a[i] - b[i]))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		return float32(1.0 / (1.0 + maxDiff))

	default:
		// Unknown comparison type – return 0.
		return 0
	}
}

type QueryOption struct {
	MaxAmount       int
	DocFilter       func(doc *Document) (store bool)
	MinimalDistance float32
	Compare         CompareVector
}

func (collection *Collection) Query(queryTexts []string, options QueryOption) (_ []*Document, err error) {
	if len(queryTexts) == 0 {
		return
	}
	// filtration
	var docs []*Document
	if options.DocFilter == nil {
		docs = collection.Documents
	} else {
		for i := range collection.Documents {
			if !options.DocFilter(collection.Documents[i]) {
				continue
			}
			docs = append(docs, collection.Documents[i])
		}
	}
	if len(docs) == 0 {
		return nil, nil
	}
	// prepare array
	type sim struct {
		doc   *Document
		value float32 // similarity
	}
	sims := make([]sim, len(docs))
	for i := range docs {
		sims[i].doc = docs[i]
	}
	// calculation
	amount := 0
	for _, queryText := range queryTexts {
		queryText = strings.TrimSpace(queryText)
		if queryText == "" {
			continue
		}
		var queryCode []float32
		queryCode, err = collection.Embed.Calculate(queryText)
		if err != nil {
			return
		}
		for i := range docs {
			sims[i].value += options.Compare.Calculate(queryCode, docs[i].Code)
		}
		amount++
	}
	// normalize
	for i := range sims {
		sims[i].value /= float32(amount)
	}
	// sort
	sort.Slice(sims, func(i, j int) bool {
		return sims[j].value < sims[i].value
	})
	// store only with acceptable distance
	var res []*Document
	for i := range sims {
		if sims[i].value < options.MinimalDistance {
			continue
		}
		res = append(res, sims[i].doc)
	}
	// filter by amount
	if 0 < options.MaxAmount && options.MaxAmount < len(res) {
		res = res[:options.MaxAmount]
	}

	return res, nil
}

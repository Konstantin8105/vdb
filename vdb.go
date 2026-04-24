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
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"time"
)

type Embeder struct {
	Model string // AI model name, e.g., "text-embedding-nomic-embed-text-v1.5"

	Endpoint string // API endpoint URL, e.g., "http://127.0.0.1:1234/v1"
	Key      string // API key for external providers (optional)

	// ContextSize int // Maximum context window size in tokens

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

	log.Printf("AI-comp endpoint: %s", endpoint)

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
	log.Printf("RouterAI response: %v", resp)
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

	code = normalizeVector(rb.Data[0].Embedding)
	return
}

type Document struct {
	ID      string
	Content string
	Code    []float32 // embedding code array

	filename string
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

type QueryOption struct {
	MaxAmount       int
	DocFilter       func(doc *Document) (store bool)
	MinimalDistance float32 // [-1...+1]
}

func (collection *Collection) Query(queryText string, options QueryOption) (_ []*Document, err error) {
	if queryText == "" {
		err = fmt.Errorf("queryText is empty")
		return
	}

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

	queryCode, err := collection.Embed.Calculate(queryText)
	if err != nil {
		return
	}

	type sim struct {
		doc   *Document
		value float32 // similarity
	}
	sims := make([]sim, len(docs))
	for i := range docs {
		sims[i].doc = docs[i]
		sims[i].value, err = dotProduct(queryCode, docs[i].Code)
		if err != nil {
			return
		}
	}

	sort.Slice(sims, func(i, j int) bool {
		return sims[i].value < sims[j].value
	})

	var res []*Document
	for i := range sims {
		if sims[i].value < options.MinimalDistance {
			continue
		}
		res = append(res, sims[i].doc)
	}

	if 0 < options.MaxAmount && options.MaxAmount < len(res) {
		res = res[:options.MaxAmount]
	}

	return res, nil
}

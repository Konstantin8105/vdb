//go:build ignore

package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"unicode"

	"github.com/Konstantin8105/vdb"
)

type Block struct {
	Filename string
	Position int
}

func (b Block) String() string {
	dat, err := json.Marshal(b)
	if err != nil {
		panic(err)
	}
	return string(dat)
}

func splitByContextTokens(filename string, tokens int) (documents []*vdb.Document) {
	data, err := os.ReadFile(filename)
	if err != nil {
		panic(err)
	}
	runes := []rune(string(data))

	blockSize := tokens        // preliminary
	intersect := blockSize / 3 // 3
	findspace := max(10, intersect/5)

	start := 0
	finish := 0
	part := 0
	for {
		finish = start + blockSize
		if len(runes) <= finish {
			finish = len(runes)
		} else {
			for range findspace {
				finish -= 1
				if unicode.IsSpace(runes[finish]) {
					break
				}
			}
		}
		body := string(runes[start:finish])
		documents = append(documents, &vdb.Document{
			ID: Block{
				Filename: filename,
				Position: part,
			}.String(),
			Content: body,
		})
		part++
		if start == finish {
			break
		}
		start += blockSize - intersect
		if len(runes) <= start {
			break
		}
		if findspace < start {
			for range findspace {
				start -= 1
				if unicode.IsSpace(runes[start]) {
					break
				}
			}
		}
	}
	return
}

func main() {
	var (
		location  = flag.String("location", "./data/", "location of source texts")
		queryText = flag.String("query", "Capital of France", "Query to RAG and acceptable a few query separate by ;")
		amount    = flag.Int("amount", 4, "amount parts from vector DB")
		reindex   = flag.Bool("reindex", false, "reindex or create a new database")
		contains  = flag.String("contains", "", "Contains strings separate by ;")
		factor    = flag.Uint("factor", 1, "contex size devided to factor for minimaze text")
	)
	flag.Parse()

	// embeder
	embed := vdb.Embeder{
		// BAD MODELS:
		// "text-embedding-nomic-embed-text-v1.5@q8_0", 2048
		//
		Model:       "text-embedding-qwen3-embedding-0.6b",
		Endpoint:    "http://127.0.0.1:1234/v1",
		Key:         "lmstudio",
		ContextSize: 10000,
	}
	// create collection if not exist
	collection, err := vdb.New("./rag/", true, embed)
	if err != nil {
		panic(err)
	}
	// reindex if need
	if *reindex {
		files, err := filepath.Glob(filepath.Join(*location, "*"))
		if err != nil {
			panic(err)
		}
		for pos, file := range files {
			docs := splitByContextTokens(file, embed.ContextSize/int(*factor))

			err = collection.AddDocuments(docs...)
			if err != nil {
				panic(err)
			}
			log.Printf("(%02d of %02d). Done: %s", pos, len(files), file)
		}
	}
	// filter
	containsFilter := strings.Split(*contains, ";")
	for i := range containsFilter {
		containsFilter[i] = strings.TrimSpace(containsFilter[i])
	}
	// search
	res, err := collection.Query(
		strings.Split(*queryText, ";"),
		vdb.QueryOption{
			MaxAmount: *amount,
			DocFilter: func(doc *vdb.Document) (store bool) {
				for _, c := range containsFilter {
					if strings.Contains(doc.Content, c) {
						store = true
					}
				}
				return
			},
			Compare: vdb.CosineSimilarity,
		})
	if err != nil {
		panic(err)
	}
	// prompt
	tmpl := template.Must(template.New("prompt").Parse(`
Summarize the document excerpts for the query "{{.Query}}" and use mandotary language of excerpts only and use only facts from excerpts.

{{range .Docs}}

Beginning of excerpt
{{.}}
End of excerpt
{{end}}
`))
	err = tmpl.Execute(os.Stdout, struct {
		Query string
		Docs  []*vdb.Document
	}{
		Query: *queryText,
		Docs:  res,
	})
	if err != nil {
		panic(err)
	}
}

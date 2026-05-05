//go:build ignore

package main

import (
	"encoding/json"
	"flag"
	"fmt"
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

	getPos := func(pos int) int {
		base := pos
		if len(runes) <= pos {
			return len(runes)
		}
		for range findspace {
			pos -= 1
			if pos < 0 {
				return 0
			}
			if runes[pos] == '\n' {
				return pos
			}
		}
		pos = base
		for range findspace {
			pos -= 1
			if pos < 0 {
				return 0
			}
			if runes[pos] == '.' {
				return pos
			}
		}
		pos = base
		for range findspace {
			pos -= 1
			if pos < 0 {
				return 0
			}
			if unicode.IsSpace(runes[pos]) {
				return pos
			}
		}
		return base
	}

	start := 0
	finish := 0
	part := 0
	for {
		finish = getPos(start + blockSize)
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
		start = getPos(start)
		if len(runes) <= start {
			break
		}
	}
	return
}

func main() {
	var (
		location  = flag.String("location", "./data/*.txt", "location of source texts")
		queryText = flag.String("query", "Capital of France", "Query to RAG and acceptable a few query separate by ;")
		amount    = flag.Int("amount", 4, "amount parts from vector DB")
		reindex   = flag.Bool("reindex", false, "reindex or create a new database")
		contains  = flag.String("contains", "", "Contains strings separate by ;")
		factor    = flag.Uint("factor", 1, "contex size devided to factor for minimaze text")
	)
	flag.Parse()

	// embeder
	embed := vdb.DefaultEmbeder()
	// vdb.Embeder{
	// 	// BAD MODELS:
	// 	// "text-embedding-nomic-embed-text-v1.5@q8_0", 2048
	// 	//
	// 	Model:       "text-embedding-qwen3-embedding-0.6b",
	// 	Endpoint:    "http://127.0.0.1:1234/v1",
	// 	Key:         "lmstudio",
	// 	ContextSize: 10000,
	// 	Dimension:   4096,
	// }
	// create collection if not exist
	collection, err := vdb.New("./rag/", true, embed)
	if err != nil {
		panic(err)
	}
	// reindex if need
	if *reindex {
		files, err := filepath.Glob(*location)
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
		if *queryText == "" {
			fmt.Fprintf(os.Stdout, "Empty query text")
			return
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
				empty := 0
				for _, c := range containsFilter {
					c := strings.TrimSpace(c)
					if c == "" {
						empty++
						continue
					}
					if strings.Contains(doc.Content, c) {
						store = true
					}
				}
				if len(containsFilter) == 0 || empty == len(containsFilter) {
					return true
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
Summarize the document excerpts for the query "{{.Query}}" and use mandotary language of excerpts only and use only facts from excerpts and also add answers on query and add links on excerpt inplace of used fact.

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

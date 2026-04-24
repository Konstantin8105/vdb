package vdb

import (
	"fmt"
	"os"
	"testing"
)

func Test(t *testing.T) {
	tb := "./testdata"
	for ie, emb := range []Embeder{
		{
			Model:    "text-embedding-nomic-embed-text-v1.5@q8_0",
			Endpoint: "http://127.0.0.1:1234/v1",
			Key:      "lmstudio",
		},
	} {
		t.Run(fmt.Sprintf("embeder_%d", ie), func(t *testing.T) {
			for _, compress := range []bool{false, true} {
				t.Run(fmt.Sprintf("compress_%v", compress), func(t *testing.T) {
					_ = os.RemoveAll(tb)
					defer func() {
						_ = os.RemoveAll(tb)
					}()
					collection, err := New(
						tb,
						compress,
						emb,
					)
					if err != nil {
						t.Fatal(err)
					}

					_, err = collection.Query("", QueryOption{})
					if err == nil {
						t.Fatalf("err is nil")
					}

					rs, err := collection.Query("one", QueryOption{DocFilter: func(doc *Document) (store bool) { return true }})
					if err != nil {
						t.Fatal(err)
					}
					if len(rs) != 0 {
						t.Fatalf("not valid amount: %d", len(rs))
					}

					err = collection.AddDocuments(nil)
					if err == nil {
						t.Fatalf("err is nil")
					}
					if len(collection.Documents) != 0 {
						t.Fatalf("not valid amount")
					}

					docs := []*Document{
						{ID: "1", Content: "number one"},
						{ID: "2", Content: "number two"},
					}
					err = collection.AddDocuments(docs...)
					if err != nil {
						t.Fatal(err)
					}
					if len(collection.Documents) != 2 {
						t.Fatalf("not valid amount")
					}

					rs, err = collection.Query("one", QueryOption{MinimalDistance: -1})
					if err != nil {
						t.Fatal(err)
					}
					if len(rs) != 2 {
						t.Fatalf("not valid amount: %d", len(rs))
					}
					rs, err = collection.Query("one", QueryOption{DocFilter: func(doc *Document) (store bool) { return true }})
					if err != nil {
						t.Fatal(err)
					}
					if len(rs) != 2 {
						t.Fatalf("not valid amount: %d", len(rs))
					}

					// reopen
					collection2, err := New(
						tb,
						compress,
						emb,
					)
					if err != nil {
						t.Fatal(err)
					}
					if len(collection2.Documents) != 2 {
						t.Fatalf("not valid amount: %d", len(collection2.Documents))
					}
					rs, err = collection.Query("one", QueryOption{MinimalDistance: -1, MaxAmount: 1})
					if err != nil {
						t.Fatal(err)
					}
					if len(rs) != 1 {
						t.Fatalf("not valid amount: %d", len(rs))
					}
				})
			}
		})
	}
}

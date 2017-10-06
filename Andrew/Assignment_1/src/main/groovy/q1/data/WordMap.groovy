package q1.data

/**
 * Created by Andrew on 10/4/2017.
 */
class WordMap {

    // Key = genre; Value is a map of words to their counts
    Map<String, GenreToWordCount> genreToWordCountMap
    Set<String> vocabulary

    WordMap() {
        this.genreToWordCountMap = [:]
        this.vocabulary = []
    }

    public WordMap addEntry(String genre, List<String> words, int documentCount = 1) {
        if (!genreToWordCountMap.containsKey(genre)) {
            genreToWordCountMap.put(genre, new GenreToWordCount(genre))
        }
        // Add each word to the count
        words.each {
            genreToWordCountMap[genre].addWord(it)
            vocabulary.add(it)
        }
        genreToWordCountMap[genre].incrementDocumentCount(documentCount)
        return this;
    }

    int getWordCount(String genre) {
        int wordCount = 0
        genreToWordCountMap[genre].wordCount.each {k, v ->
            wordCount += v
        }
        return wordCount
    }

    int getWordCount(String genre, String word) {
        return genreToWordCountMap[genre].wordCount[word] ?: 0
    }

    int getVocabularyWordCount() {
        return vocabulary.size()
    }

    public getDocumentCount(String genre = null) {
        int count = 0
        genreToWordCountMap.each {k, v ->
            if (genre == null) {
                count += v.documentCount
            } else if (k.equals(genre)) {
                count += v.documentCount
            }
        }
        return count
    }

    /**
     * Our data model is such that adding new words might not inform every genre of the existence of that word.
     * In order to properly smooth, every genre must be aware of the entire vocabulary. This adds each word in vocabulary
     * into each genre if it doesn't yet exist.
     *
     */
    public void synchronizeVocabulary() {
        // Make each genre acknowledge each vocab word
        vocabulary.each {String vocabWord ->
            genreToWordCountMap.each { String genre, GenreToWordCount genreToWordCountMap ->
                genreToWordCountMap.acknowledgeWord(vocabWord)
            }
        }
    }

    private class GenreToWordCount {
        String genre;
        int documentCount
        Map<String, Integer> wordCount

        GenreToWordCount(String genre) {
            this.genre = genre
            wordCount = [:]
            documentCount = 0
        }

        void addWord(String word) {
            if (wordCount.containsKey(word)) {
                wordCount[word] = wordCount[word] + 1
            } else {
                wordCount[word] = 1
            }
        }

        // sets word count to 0 for word, if it's not in the map
        void acknowledgeWord(String word) {
            if (!wordCount.containsKey(word)) {
                wordCount.put(word, 0)
            }
        }

        public void incrementDocumentCount(int count) {
            documentCount += count
        }
    }
}

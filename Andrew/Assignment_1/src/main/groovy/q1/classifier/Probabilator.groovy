package q1.classifier

import q1.data.WordMap

/**
 * Created by Andrew on 10/5/2017.
 */
class Probabilator {

    WordMap wordMap

    Probabilator(WordMap wordMap) {
        this.wordMap = wordMap
    }

    private float getProbabilityOfWord(String genre, String word) {
        int wordCountInGenre = wordMap.getWordCount(genre, word)
        int totalGenreWordCount = wordMap.getWordCount(genre)
        int vocabularyWordCount = wordMap.getVocabularyWordCount()

        return (wordCountInGenre + 1) / (totalGenreWordCount + vocabularyWordCount)
    }

    private float getProbabilityOfGenre(String genre) {
        int genreDocumentCount = wordMap.getDocumentCount(genre)
        int totalDocumentCount = wordMap.getDocumentCount()

        return genreDocumentCount / totalDocumentCount
    }

    /**
     * Given a jumble of words representing a document, get the probabilities of that document belonging
     * to each genre
     */
    public ProbabilatorResult classifyWords(List<String> words) {
        ProbabilatorResult result = new ProbabilatorResult()

        // for each genre
        wordMap.genreToWordCountMap.each {genre, v ->
            // calculate its probability of this document belonging to it
            float p = getProbabilityOfGenre(genre)
            words.each {word ->
                p *= getProbabilityOfWord(genre, word)
            }
            result.genreToProbabilityMap.put(genre, p)
        }

        return result
    }

    class ProbabilatorResult {
        Map<String, Float> genreToProbabilityMap

        ProbabilatorResult() {
            genreToProbabilityMap = [:]
        }

        public String getHighestProbableGenre() {
            String result = "";
            float highestCurrentProbability = 0f
            genreToProbabilityMap.each {k, v ->
                if (v > highestCurrentProbability) {
                    result = k
                }
            }
            return result;
        }
    }
}

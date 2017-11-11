package q1.parser

import q1.data.WordMap

/**
 * Created by Andrew on 10/4/2017.
 */
class TrainingParser {

    String filename;
    private final File file;

    TrainingParser(String filename) {
        this.filename = filename;
        file = new File(filename)
    }

    WordMap parseFile() {
        WordMap result = new WordMap()
        file.eachLine {
            ParsedLine parsedLine = new ParsedLine(it)
            result.addEntry(parsedLine.genre, parsedLine.lineContent)
        }
        result.synchronizeVocabulary()
        return result
    }

    class ParsedLine {
        List<String> lineContent
        String genre

        ParsedLine(String rawString) {
            String s = rawString.replaceAll("\\s","") // remove all whitespace
            List<String> lineContentsAndGenre = s.tokenize(',') // delimit on commas
            List<String> lastLineContentAndGenre = lineContentsAndGenre.last().tokenize(':') // delimit final word from its genre

            this.genre = lastLineContentAndGenre.last()
            this.lineContent = lineContentsAndGenre.init() // get all but last element
            this.lineContent += lastLineContentAndGenre.first() // add the last element
        }
    }
}

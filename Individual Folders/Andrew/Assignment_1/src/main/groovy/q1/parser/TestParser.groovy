package q1.parser

/**
 * Created by Andrew on 10/5/2017.
 */
class TestParser {

    String filename;
    private final File file;

    TestParser(String filename) {
        this.filename = filename;
        file = new File(filename)
    }

    public List<List<String>> getDocumentsAsBagsOfWords() {
        List<List<String>> result = []
        file.eachLine {
            ParsedLine parsedLine = new ParsedLine(it)
            result.add(parsedLine.lineContent)
        }
        return result
    }

    class ParsedLine {
        List<String> lineContent

        ParsedLine(String rawString) {
            String s = rawString.replaceAll("\\s","") // remove all whitespace
            this.lineContent = s.tokenize(',') // delimit on commas
        }
    }
}

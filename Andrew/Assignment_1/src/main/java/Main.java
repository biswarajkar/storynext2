import q1.classifier.Probabilator;
import q1.parser.TestParser;
import q1.parser.TrainingParser;

import java.util.List;

/**
 * Created by Andrew on 10/4/2017.
 */
public class Main {

    public static void main(String[] args) {
        String trainingDataFilename = args[0];
        String testDataFileName = args[1];

        TrainingParser trainingParser = new TrainingParser(trainingDataFilename);
        TestParser testParser = new TestParser(testDataFileName);

        Probabilator probabilator = new Probabilator(trainingParser.parseFile());

        List<List<String>> testData = testParser.getDocumentsAsBagsOfWords();

        for (int i = 0; i < testData.size(); i++) {
            List<String> document = testData.get(i);
            System.out.println("TEST " + (i + 1) + ": Classifying document '" + list2String(document) + "'");

            Probabilator.ProbabilatorResult result = probabilator.classifyWords(document);

            String resultGenre = result.getHighestProbableGenre();
            System.out.println("Highest probable genre: '" + resultGenre + "'");
            System.out.println("with probability " + result.getGenreToProbabilityMap().get(resultGenre));
            System.out.println();

        }
    }

    private static String list2String(List<String> in) {
        String result = "";
        for (String s : in) {
            result += s + ", ";
        }
        return result.substring(0, result.length() - 2);
    }
}

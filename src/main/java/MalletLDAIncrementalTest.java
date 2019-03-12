import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.types.Alphabet;
import cc.mallet.types.IDSorter;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.Iterator;
import java.util.Locale;
import java.util.TreeSet;
import java.util.regex.Pattern;

/**
 * @author Victor Oliveira
 */
public class MalletLDAIncrementalTest {

	private static String TOPIC_MODEL_FILE_PATH = "data/my-previous-model.dat";
	private static String CORPUS_1_FILE_PATH = "data/kt1.txt";
	private static String CORPUS_2_FILE_PATH = "data/kt2.txt";

	private static int NUMBER_OF_TERMS_BY_TOPIC = 10;
	private static int NUMBER_OF_TOPICS = 10;
	private static int NUMBER_OF_ITERATIONS = 2000;
	private static int NUMBER_OF_THREADS = 2;

	private static double LDA_ALPHA_SUM = 1.0;
	private static double LDA_BETA = 0.01;

	public static void main(String[] args) throws Exception {

		if (true) {
			PrintStream fileStream = new PrintStream("data/output-kt-total.txt");
			System.setOut(fileStream);
		}

		File topicModuleFile = new File(TOPIC_MODEL_FILE_PATH);

		ParallelTopicModel model = null;
		String fileName = null;

		if (topicModuleFile.exists()) {
			model = ParallelTopicModel.read(topicModuleFile);
			fileName = CORPUS_2_FILE_PATH;
		} else {
			fileName = CORPUS_1_FILE_PATH;
			model = new ParallelTopicModel(NUMBER_OF_TOPICS, LDA_ALPHA_SUM, LDA_BETA);
			model.setNumThreads(NUMBER_OF_THREADS);
			model.setNumIterations(NUMBER_OF_ITERATIONS);
		}

		ArrayList<Pipe> pipeList = new ArrayList<>();

		pipeList.add( new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")) );
		pipeList.add( new TokenSequence2FeatureSequence() );

		InputStreamReader fileReader = new InputStreamReader(new FileInputStream(fileName), StandardCharsets.UTF_8);

		Iterator<Instance> instanceIterator = new CsvIterator(fileReader,
			Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"), 3, 2, 1);

		InstanceList training = new InstanceList(new SerialPipes(pipeList));
		training.addThruPipe(instanceIterator); // data, label, name fields

		model.addInstances(training);
		model.estimate();

		Alphabet dataAlphabet = training.getDataAlphabet();
		double[] topicDistribution = model.getTopicProbabilities(0);
		// Get an array of sorted sets of word ID/count pairs
		ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();

		// Show top 5 words in topics with proportions for the first document
		for (int topic = 0; topic < NUMBER_OF_TOPICS; topic++) {
			Iterator<IDSorter> iterator = topicSortedWords.get(topic).iterator();

			Formatter out = new Formatter(new StringBuilder(), Locale.US);
			out.format("%d\t%.3f\t", topic, topicDistribution[topic]);
			int rank = 0;
			while (iterator.hasNext() && rank < NUMBER_OF_TERMS_BY_TOPIC) {
				IDSorter idCountPair = iterator.next();
				out.format("%s (%.0f) ", dataAlphabet.lookupObject(idCountPair.getID()), idCountPair.getWeight());
				rank++;
			}

			System.out.println(out);
		}

		model.write(new File(TOPIC_MODEL_FILE_PATH));
	}
}

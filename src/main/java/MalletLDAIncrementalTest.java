import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Properties;
import java.util.regex.Pattern;

/**
 * @author Victor Oliveira
 */
public class MalletLDAIncrementalTest {


	public static void main(String[] args) throws Exception {
		String topicModelFilePath = props.getProperty("topic-model-file");
		File topicModuleFile = new File(topicModelFilePath);
		String fileName = null;
		ParallelTopicModel model = null;

		if (topicModuleFile.exists()) {

			model = ParallelTopicModel.read(topicModuleFile);
			fileName = props.getProperty("corpus2-file-path");

		} else {

			fileName = props.getProperty("corpus1-file-path");

			int numberOfTopics = Integer.parseInt(props.getProperty("number-of-topics"));
			int numberOfThreads = Integer.parseInt(props.getProperty("number-of-threads"));
			int numberOfIterations = Integer.parseInt(props.getProperty("number-of-iterations"));

			model = new ParallelTopicModel(numberOfTopics, 1.0, 0.01);
			model.setNumThreads(numberOfThreads);
			model.setNumIterations(numberOfIterations);
		}

		ArrayList<Pipe> pipeList = new ArrayList<>();

		pipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
		pipeList.add(new TokenSequence2FeatureSequence());

		InputStreamReader fileReader =
			new InputStreamReader(new FileInputStream(fileName), StandardCharsets.UTF_8);

		Iterator<Instance> instanceIterator = new CsvIterator(fileReader,
			Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"), 3, 2, 1);

		InstanceList training = new InstanceList(new SerialPipes(pipeList));
		training.addThruPipe(instanceIterator); // data, label, terms

		model.addInstances(training);
		model.estimate();
		model.write(topicModuleFile);

		printAllReports(model);
	}

	private static void printAllReports(ParallelTopicModel topicModel) throws IOException {
		int numberOfTermsByTopic = Integer.parseInt(props.getProperty("number-of-terms-by-topic"));

		String topicsFilePath = props.getProperty("output-topics-file");
		String topicsXMLReportFilePath = props.getProperty("output-topic-xml-report-file");
		String topicPhraseXMLReportFilePath = props.getProperty("output-topic-phrase-xml-report-file");
		String documentTopicsFilePath = props.getProperty("output-document-topics-file");
		String topicWordWeightFilePath = props.getProperty("output-topic-word-weight-file");
		String topicTypeTopicCountsFilePath = props.getProperty("output-topic-type-topic-counts-file");

		if (topicsFilePath != null) {
			topicModel.printTopWords(new File(topicsFilePath), numberOfTermsByTopic, false);
		}

		File malletDir = new File("data/mallet");
		malletDir.mkdirs();

		if (topicsXMLReportFilePath != null) {
			PrintWriter out = new PrintWriter(topicsXMLReportFilePath);
			topicModel.topicXMLReport(out, numberOfTermsByTopic);
			out.close();
		}

		if (topicPhraseXMLReportFilePath != null) {
			PrintWriter out = new PrintWriter(topicPhraseXMLReportFilePath);
			topicModel.topicPhraseXMLReport(out, numberOfTermsByTopic);
			out.close();
		}

		if (documentTopicsFilePath != null) {
			PrintWriter out3 = new PrintWriter(new FileWriter(documentTopicsFilePath));
			topicModel.printDocumentTopics(out3, 0.0, -1);
			out3.close();
		}

		if (topicWordWeightFilePath != null) {
			topicModel.printTopicWordWeights(new File(topicWordWeightFilePath));
		}

		if (topicTypeTopicCountsFilePath != null) {
			topicModel.printTypeTopicCounts(new File(topicTypeTopicCountsFilePath));
		}
	}

	private static Properties loadProperties() {
		Properties props = new Properties();
		InputStream input = null;
		ClassLoader loader = Thread.currentThread().getContextClassLoader();

		try {
			String filename = "mallet.properties";
			input = loader.getResourceAsStream(filename);
			props.load(input);
		} catch (Exception ignored) {
		} finally {
			try {
				if (input != null) input.close();
			} catch (Exception ignored) {
			}
		}

		return props;
	}

	private static Properties props = loadProperties();
}
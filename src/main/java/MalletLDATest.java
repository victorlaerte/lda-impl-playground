import cc.mallet.topics.LDA;
import cc.mallet.types.InstanceList;
import cc.mallet.util.Randoms;
import java.io.File;
import java.io.IOException;

/**
 * @author Victor Oliveira
 */
public class MalletLDATest {

	public static void main(String[] args) throws IOException {

		String path = "data/kotlin_sample.txt";

		File file = new File(path);

		if (file.exists()) {
			InstanceList instanceList = InstanceList.load(file);

			int numIterations = 1000;
			int numTopWords = 20;

			LDA lda = new LDA(10);
			lda.estimate(instanceList, numIterations, 50, 0, null,
				new Randoms());  // should be 1100
			lda.printTopWords(numTopWords, true);
			lda.printDocumentTopics(new File(path + ".lda"));
		}
	}
}

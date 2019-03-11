import models.GibbsSamplingLDA;

/**
 * @author Marcellus Tavares
 */
public class JLDADMMTest {

	public static void main(String[] args) throws Exception {

		//	new GibbsSamplingLDA(cmdArgs.corpus,
		//		cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
		//		cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
		//		cmdArgs.initTopicAssgns, cmdArgs.savestep);


		GibbsSamplingLDA lda = new GibbsSamplingLDA(
			"data/kotlin_sample.txt",
			20, .1, .1,
			2000, 20, "model");

		lda.inference();

	}
}

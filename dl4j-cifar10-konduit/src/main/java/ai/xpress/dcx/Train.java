package ai.xpress.dcx;

import org.apache.commons.lang3.time.StopWatch;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class Train {
    protected static final Logger log = LoggerFactory.getLogger(Train.class);

    private static final int height = 32;
    private static final int width = 32;
    private static final int channels = 3;
    private static final int numLabels = CifarLoader.NUM_LABELS;
    private static final int batchSize = 128;
    private static final long seed = 42;
    private static final int epochs = 30;

    public static void main(String... args) throws Exception {
        log.info("Starting CIFAR10 Training.");

        log.info("Loading CIFAR10 Data.");


        var normalizer = new NormalizerStandardize();

        var cifarTrain = new Cifar10DataSetIterator(
                batchSize,
                new int[]{height, width},
                DataSetType.TRAIN,
                null,
                seed
        );

        log.info("Fitting normalizer");
        normalizer.fit(cifarTrain);

        cifarTrain.setPreProcessor(normalizer);

        var cifarEval = new Cifar10DataSetIterator(
                8,
                new int[]{height, width},
                DataSetType.TEST,
                null,
                seed
        );
        cifarEval.setPreProcessor(normalizer);

        var criterion = LossFunctions.LossFunction.MCXENT;
        var optim = new Adam();

        var model = new Model();
        model.compile(optim, criterion);

        model.addListeners(
                new ScoreIterationListener(100),
                new EvaluativeListener(cifarEval, 5, InvocationType.EPOCH_END)
        );

        var stopWatch = StopWatch.createStarted();
        model.fit(cifarTrain, epochs);
        stopWatch.stop();

        var eval = model.evaluate(cifarEval);
        log.info("Evaluation: {}", eval.toString());

        log.info("Training finished.  Took {} seconds.", stopWatch.getTime(TimeUnit.SECONDS));
        model.save(new File("model.dl4j"));
        NormalizerSerializer.getDefault().write(normalizer, new File("model.normalizer.dl4j"));
    }


    static class Model extends Module {
        private Layer conv1 = new Convolution2D.Builder(5, 5).nIn(3).nOut(6).build();
        private Layer pool = new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).build();
        private Layer conv2 = new Convolution2D.Builder(5, 5).nIn(6).nOut(16).build();
        private Layer fc1 = new DenseLayer.Builder().nIn(16 * 5 * 5).nOut(120).build();
        private Layer fc2 = new DenseLayer.Builder().nIn(120).nOut(84).build();
        private Layer fc3 = new DenseLayer.Builder().nIn(84).nOut(10).build();
        private Layer relu = new ActivationLayer(Activation.RELU);
        private Layer softmax = new ActivationLayer(Activation.SOFTMAX);

        public NeuralNetConfiguration.ListBuilder forward(NeuralNetConfiguration.ListBuilder x) {
            x.layer(conv1);
            x.layer(pool);
            x.layer(relu);
            x.layer(conv2);
            x.layer(pool);
            x.layer(relu);
            x.layer(fc1);
            x.layer(relu);
            x.layer(fc2);
            x.layer(fc3);
            x.layer(softmax);
            x.setInputType(InputType.convolutional(height, width, channels));
            return x;
        }
    }

    static abstract class Module extends NeuralNetConfiguration {
        private MultiLayerNetwork model;

        abstract protected NeuralNetConfiguration.ListBuilder forward(NeuralNetConfiguration.ListBuilder in);

        public void compile(IUpdater optimizer, LossFunctions.LossFunction loss) {
            compile(optimizer, loss.getILossFunction());
        }

        public void compile(IUpdater optimizer, ILossFunction loss) {
            var conf = new NeuralNetConfiguration.Builder()
                    .seed(42)
                    .updater(optimizer);
            var layers = forward(conf.list());
            layers.layer(new LossLayer.Builder(loss).build());

            model = new MultiLayerNetwork(layers.build());
            model.init();
        }

        public MultiLayerNetwork getModel() {
            return model;
        }

        public void addListeners(TrainingListener... listeners) {
            if (model != null) {
                model.addListeners(listeners);
            }
        }

        public void fit(DataSetIterator dsi, int epochCount) {
            if (model != null) {
                model.fit(dsi, epochCount);
            }
        }

        public void save(File f) throws IOException {
            if (model != null) {
                model.save(f, true);
            }
        }

        public void save(File f, boolean saveUpdater) throws IOException {
            if (model != null) {
                model.save(f, saveUpdater);
            }
        }

        public Evaluation evaluate(DataSetIterator dsi) {
            if (model != null) {
                return model.evaluate(dsi);
            }
            return null;
        }
    }
}

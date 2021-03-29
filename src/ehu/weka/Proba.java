package ehu.weka;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;
import java.util.HashSet;

import com.sun.xml.bind.v2.util.DataSourceSource;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class Proba {

	public static void main(String[] args) {


            ConverterUtils.DataSource loader;
			try {
				loader = new ConverterUtils.DataSource("data_supervised.arff");
	            Instances data = loader.getDataSet();
	            data.setClassIndex(data.numAttributes()-1);

	            
	            AttributeSelection as = new AttributeSelection();
	            as.setSearch(new BestFirst());
	            as.setEvaluator(new CfsSubsetEval());
	            as.SelectAttributes(data);
	            int[] indices = as.selectedAttributes();
	            Remove remove = new Remove();
	            remove.setAttributeIndicesArray(indices);
	            remove.setInvertSelection(true);
	            remove.setInputFormat(data);
	            Instances newData = Filter.useFilter(data, remove);
	            
	            //orain hold out egingo dugu
	            
	            Randomize randomize = new Randomize();
	            randomize.setInputFormat(data);
	            Instances dataRandom = Filter.useFilter(data, randomize);
	            
	            
	            RemovePercentage rmp = new RemovePercentage();
	            rmp.setInputFormat(dataRandom);
	            rmp.setPercentage(30);
	            Instances train = Filter.useFilter(dataRandom, rmp);
	            rmp = new RemovePercentage();
	            rmp.setInputFormat(dataRandom);
	            rmp.setPercentage(30);
	            rmp.setInvertSelection(true);
	            Instances test = Filter.useFilter(dataRandom, rmp);

	            NaiveBayes nbproba = new NaiveBayes();
	            nbproba.buildClassifier(train);
	            Evaluation proba = new Evaluation(test);
	            proba.evaluateModel(nbproba, test);
	            System.out.println("Probetan hurrengo F-Score lortu da:");
	            System.out.println(proba.weightedFMeasure());
	            
	            
	            
	            
	            NaiveBayes nb = new NaiveBayes();
	            nb.buildClassifier(newData);
	            
	            

	            System.out.println("Missing values-ak mantenduz:");
				loader = new ConverterUtils.DataSource("data_test_blind.arff");
	            test = loader.getDataSet();
	            test.setClassIndex(test.numAttributes()-1);
	            remove = new Remove();
	            remove.setAttributeIndicesArray(indices);
	            remove.setInvertSelection(true);
	            remove.setInputFormat(test);
	            Instances newTest = Filter.useFilter(test, remove);
	            for (int k = 0; k<newTest.numAttributes()-1; k++) {
	            	System.out.println(newTest.attribute(k).name());
	            	System.out.println(newTest.attributeStats(k).missingCount + " missing values");
	            }
	            Evaluation eval = new Evaluation(newTest);
	            eval.evaluateModel(nb, newTest);

	            
	            //orain missing values-ak aldatuko ditut
	            System.out.println("Missing values moda eta medianekin aldatuz:");
				loader = new ConverterUtils.DataSource("data_test_blind.arff");
	            test = loader.getDataSet();
	            test.setClassIndex(test.numAttributes()-1);
	            remove = new Remove();
	            remove.setAttributeIndicesArray(indices
	            		);
	            remove.setInvertSelection(true);
	            remove.setInputFormat(test);
	            newTest = Filter.useFilter(test, remove);
	            ReplaceMissingValues rmv = new ReplaceMissingValues();
	            rmv.setInputFormat(newTest);
	            newTest = Filter.useFilter(newTest, rmv);
	            
	            
	            for (int k = 0; k<newTest.numAttributes()-1; k++) {
	            	System.out.println(newTest.attribute(k).name());
	            	System.out.println(newTest.attributeStats(k).missingCount + " missing values");
	            }
	            Evaluation evalB = new Evaluation(newTest);
	            evalB.evaluateModel(nb, newTest);
	            
	            
	            
	            System.out.println("Eta orain ea iragaparen bat aldatu den ala ez ikusiko dugu");
	            
	            for(int i  =0; i<eval.predictions().size()-1;i++) {
	            	System.out.println(i + ". balioa lehen: " + eval.predictions().get(i).predicted()+ " eta balio berria: "+evalB.predictions().get(i).predicted());
	            }
	            
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}


            
	}

	
}

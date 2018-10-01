package core;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.StructType;

public class FeatureExtraction
{
	private static SparkSession spark;
	
	private static StructType sch = new StructType().add("timeStamp", "float").add("activityID", "int").add("heartRate", "int")
			.add("handTemp", "float").add("handAcc16x", "float").add("handAcc16y", "float")
			.add("handAcc16z", "float").add("handAcc6x", "float").add("handAcc6y", "float")
			.add("handAcc6z", "float").add("handGyrox", "float").add("handGyroy", "float").add("handGyroz", "float")
			.add("handMagx", "float").add("handMagy", "float").add("handMagz", "float").add("handOrx", "float")
			.add("handOry", "float").add("handOrz", "float").add("chestTemp", "float").add("chestAcc16x", "float")
			.add("chestAcc16y", "float").add("chestAcc16z", "float").add("chestAcc6x", "float")
			.add("chestAcc6y", "float").add("chestAcc6z", "float").add("chestGyrox", "float")
			.add("chestGyroy", "float").add("chestGyroz", "float").add("chestMagx", "float")
			.add("chestMagy", "float").add("chestMagz", "float").add("chestOrx", "float").add("chestOry", "float")
			.add("chestOrz", "float").add("ankleTemp", "float").add("ankleAcc16x", "float")
			.add("ankleAcc16y", "float").add("ankleAcc16z", "float").add("ankleAcc6x", "float")
			.add("ankleAcc6y", "float").add("ankleAcc6z", "float").add("ankleGyrox", "float")
			.add("ankleGyroy", "float").add("ankleGyroz", "float").add("ankleMagx", "float")
			.add("ankleMagy", "float").add("ankleMagz", "float").add("ankleOrx", "float").add("ankleOry", "float")
			.add("ankleOrz", "float").add("unknown1", "float").add("unknown2", "float").add("unknown3", "float");
	
	private static StructType featuresSchema = new StructType().add("label", "int").add("maxHR", "int")
			.add("maxHtemp", "float").add("maxCtemp", "float").add("maxAtemp", "float")
			.add("corrxyHandAcc", "double").add("corryzHandAcc", "double")
			.add("corrxzHandAcc", "double").add("corrxyHandMag", "double").add("corryzHandMag", "double")
			.add("corrxzHandMag", "double").add("corrxyChestAcc", "double").add("corryzChestAcc", "double")
			.add("corrxzChestAcc", "double").add("corrxyChestMag", "double").add("corryzChestMag", "double")
			.add("corrxzChestMag", "double").add("corrxyAnkleAcc", "double").add("corryzAnkleAcc", "double")
			.add("corrxzAnkleAcc", "double").add("corrxyAnkleMag", "double").add("corryzAnkleMag", "double")
			.add("corrxzAnkleMag", "double");
			
	public static void main(String[] args) throws IOException
	{
		spark = SparkSession.builder().appName("Feature Extraction").getOrCreate();
		
		Dataset<Row> d1 = processOneFile(sch, "/tmp/sub1.txt");
		//Dataset<Row> d2 = processOneFile(sch, "/tmp/sub2.txt");
		//Dataset<Row> d3 = processOneFile(sch, "/tmp/sub3.txt");
		//Dataset<Row> d4 = processOneFile(sch, "/tmp/sub4.txt");
		//Dataset<Row> d6 = processOneFile(sch, "/tmp/sub6.txt");
		//Dataset<Row> d7 = processOneFile(sch, "/tmp/sub7.txt");
		//Dataset<Row> d8 = processOneFile(sch, "/tmp/sub8.txt");
		
		//Dataset<Row> unionData = d1.union(d2).union(d3).union(d4).union(d6).union(d7).union(d7);
		//unionData.repartition(1).write().option("header", "true").csv("/tmp/features");
		d1.repartition(1).write().option("header", "true").csv("/tmp/features-sub1");
	}
	
	private static Dataset<Row> processOneFile(StructType schema, String fileName)
	{
		Dataset<Row> ds = spark.read().option("header", "false").option("sep", ",").schema(schema).csv(fileName);
		
		String viewName = "sensorData";
		ds.createOrReplaceTempView(viewName);
		ds.cache();

		Dataset<Row> subsetColumnsData = spark.sql(
				"select timeStamp, activityID, heartRate, handTemp, handAcc16x, handAcc16y, handAcc16z, handMagx, handMagy, handMagz,"
						+ "chestTemp, chestAcc16x, chestAcc16y, chestAcc16z, chestMagx, chestMagy, chestMagz,"
						+ "ankleTemp, ankleAcc16x, ankleAcc16y, ankleAcc16z, ankleMagx, ankleMagy, ankleMagz from sensorData");

		String filteredData = "filteredData";
		subsetColumnsData.createOrReplaceTempView(filteredData);
		subsetColumnsData.cache();
		
		Dataset<Row> updatedData = updateLabels(filteredData);
		updatedData.createOrReplaceTempView("updatedData");
		updatedData.cache();

		Dataset<Row> subset1 = spark.sql("select * from updatedData where label = 1");
		Dataset<Row> subset2 = spark.sql("select * from updatedData where label = 2");
		Dataset<Row> subset3 = spark.sql("select * from updatedData where label = 3");

		Dataset<Row> newFeaturesDF1 = calculateNewFeatures(subset1);
		Dataset<Row> newFeaturesDF2 = calculateNewFeatures(subset2);
		Dataset<Row> newFeaturesDF3 = calculateNewFeatures(subset3);
		
		return newFeaturesDF1.union(newFeaturesDF2).union(newFeaturesDF3);
	}

	private static Dataset<Row> updateLabels(String datasetViewName)
	{
		String command1 = String.format("select * from %s where activityID = 1", datasetViewName);
		Dataset<Row> values1 = spark.sql(command1);
		values1 = values1.withColumn("label", functions.lit(1)).drop("activityID");

		String command2 = String.format("select * from %s where activityID = 2", datasetViewName);
		Dataset<Row> values2 = spark.sql(command2);
		values2 = values2.withColumn("label", functions.lit(1)).drop("activityID");

		String command3 = String.format("select * from %s where activityID = 3", datasetViewName);
		Dataset<Row> values3 = spark.sql(command3);
		values3 = values3.withColumn("label", functions.lit(1)).drop("activityID");

		String command17 = String.format("select * from %s where activityID = 17", datasetViewName);
		Dataset<Row> values17 = spark.sql(command17);
		values17 = values17.withColumn("label", functions.lit(1)).drop("activityID");

		String command16 = String.format("select * from %s where activityID = 16", datasetViewName);
		Dataset<Row> values16 = spark.sql(command16);
		values16 = values16.withColumn("label", functions.lit(2)).drop("activityID");

		String command13 = String.format("select * from %s where activityID = 13", datasetViewName);
		Dataset<Row> values13 = spark.sql(command13);
		values13 = values13.withColumn("label", functions.lit(2)).drop("activityID");

		String command4 = String.format("select * from %s where activityID = 4", datasetViewName);
		Dataset<Row> values4 = spark.sql(command4);
		values4 = values4.withColumn("label", functions.lit(2)).drop("activityID");

		String command7 = String.format("select * from %s where activityID = 7", datasetViewName);
		Dataset<Row> values7 = spark.sql(command7);
		values7 = values7.withColumn("label", functions.lit(2)).drop("activityID");

		String command6 = String.format("select * from %s where activityID = 6", datasetViewName);
		Dataset<Row> values6 = spark.sql(command6);
		values6 = values6.withColumn("label", functions.lit(2)).drop("activityID");

		String command12 = String.format("select * from %s where activityID = 12", datasetViewName);
		Dataset<Row> values12 = spark.sql(command12);
		values12 = values12.withColumn("label", functions.lit(3)).drop("activityID");

		String command5 = String.format("select * from %s where activityID = 5", datasetViewName);
		Dataset<Row> values5 = spark.sql(command5);
		values5 = values5.withColumn("label", functions.lit(3)).drop("activityID");

		String command24 = String.format("select * from %s where activityID = 24", datasetViewName);
		Dataset<Row> values24 = spark.sql(command24);
		values24 = values24.withColumn("label", functions.lit(3)).drop("activityID");

		Dataset<Row> result = values1.union(values2).union(values3).union(values17).union(values16).union(values13)
				.union(values4).union(values7).union(values6).union(values12).union(values5).union(values24);

		return result;
	}

	private static Dataset<Row> calculateNewFeatures(Dataset<Row> inputData)
	{
		inputData = inputData.sort(functions.asc("timeStamp"));		
		inputData.createOrReplaceTempView("input");
		inputData.cache();
		
		Dataset<Row> lastTimestampDS = spark.sql("select max(timeStamp) from input");
		float lastTimestamp = lastTimestampDS.head().getFloat(0);
		
		float startTimestamp = inputData.select("timeStamp").head().getFloat(0);
		float endTimestamp = (float) (startTimestamp + 5);

		List<Row> data = new ArrayList<>();		

		while (startTimestamp < lastTimestamp)
		{
			String query = String.format("select * from input where timeStamp >= %s and timeStamp < %s", startTimestamp, endTimestamp);
			Dataset<Row> window = spark.sql(query);
			window.createOrReplaceTempView("window");			
			window.cache();
			
			if (window.count() > 100)
			{
				Dataset<Row> maxHRDS = spark.sql("select max(heartRate) from window");
				int maxHeartRate = maxHRDS.head().getInt(0);

				Dataset<Row> labelDS = spark.sql("select label from window");
				int label = labelDS.head().getInt(0);
				
				Dataset<Row> maxHandTempDS = spark.sql("select max(handTemp) from window");
				float maxHandTemp = maxHandTempDS.head().getFloat(0);
				
				Dataset<Row> maxChestTempDS = spark.sql("select max(chestTemp) from window");
				float maxChestTemp = maxChestTempDS.head().getFloat(0);
				
				Dataset<Row> maxAnkleTempDS = spark.sql("select max(ankleTemp) from window");
				float maxAnkleTemp = maxAnkleTempDS.head().getFloat(0);
				
				double corrxyHandAcc =  window.stat().corr("handAcc16x", "handAcc16y");
				double corryzHandAcc =  window.stat().corr("handAcc16y", "handAcc16z");
				double corrxzHandAcc =  window.stat().corr("handAcc16x", "handAcc16z");
				double corrxyHandMag =  window.stat().corr("handMagx", "handMagy");
				double corryzHandMag =  window.stat().corr("handMagy", "handMagz");
				double corrxzHandMag =  window.stat().corr("handMagx", "handMagz");
				
				double corrxyChestAcc = window.stat().corr("chestAcc16x", "chestAcc16y");
				double corryzChestAcc = window.stat().corr("chestAcc16y", "chestAcc16z");
				double corrxzChestAcc = window.stat().corr("chestAcc16x", "chestAcc16z");
				double corrxyChestMag = window.stat().corr("chestMagx", "chestMagy");
				double corryzChestMag = window.stat().corr("chestMagy", "chestMagz");
				double corrxzChestMag = window.stat().corr("chestMagx", "chestMagz");
				
				double corrxyAnkleAcc = window.stat().corr("ankleAcc16x", "ankleAcc16y");
				double corryzAnkleAcc = window.stat().corr("ankleAcc16y", "ankleAcc16z");
				double corrxzAnkleAcc = window.stat().corr("ankleAcc16x", "ankleAcc16z");
				double corrxyAnkleMag = window.stat().corr("ankleMagx", "ankleMagy");
				double corryzAnkleMag = window.stat().corr("ankleMagy", "ankleMagz");
				double corrxzAnkleMag = window.stat().corr("ankleMagx", "ankleMagz");
				
				Row row = RowFactory.create(label, maxHeartRate, maxHandTemp, maxChestTemp, maxAnkleTemp,
						corrxyHandAcc, corryzHandAcc, corrxzHandAcc, corrxyHandMag, corryzHandMag, corrxzHandMag,
						corrxyChestAcc, corryzChestAcc, corrxzChestAcc, corrxyChestMag, corryzChestMag, corrxzChestMag,
						corrxyAnkleAcc, corryzAnkleAcc, corrxzAnkleAcc, corrxyAnkleMag, corryzAnkleMag, corrxzAnkleMag);
				data.add(row);
			}

			startTimestamp += 1;
			endTimestamp += 1;
		}

		return spark.createDataFrame(data, featuresSchema);
	}
}
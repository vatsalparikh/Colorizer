/*
================================================================================================================
Project - Colorization

Class for Colorization Assignment.
================================================================================================================
*/

package app;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Colorization {
	
	//total input rows: 48894
	static int trainingStart = 0;
	static int trainingEnd = (int)(48894*0.8);
	static int testStart = trainingEnd + 1;
			
	static int testEnd = testStart + (int)(48894*0.15);
	static int validationStart = testEnd + 1;
	static int validationEnd = 48893;
	static int TOTAL_LINES = 48894;

	static int numClusters = 40;
	static int maxReclassification = 40;
	static int[] bwCluster;
	static int[] colorCluster;
	static int[][] inBwclusters = new int[numClusters][9];
	static int[][] inColorclusters = new int[numClusters][3];
	
	
	//static int MAX_INTENSITY = 255;
	//static int CLUSTERS = 10;
	static String IN_COLOR_PATH = "src//data//color.csv";
	static String IN_BW_PATH = "src//data//input.csv";

	static String OUT_BW_PATH = "src//data//input.csv";
	static String OUT_COLOR_PATH = "src//data//output11.csv"; //output color csv
	static HashMap<Integer, ArrayList<Integer>> inBwMap = new HashMap<Integer, ArrayList<Integer>>(); // Map for BlackWhite data - (Key,Value) as (i, Cell<0 to 8>)
	static HashMap<Integer, ArrayList<Integer>> inColorMap = new HashMap<Integer, ArrayList<Integer>>(); // Map for Color data - (Key,Value) as (i, <R,G,B>)
	
	static HashMap<Integer, ArrayList<Integer>> clusterBwMap = new HashMap<Integer, ArrayList<Integer>>();
	static HashMap<Integer, ArrayList<Integer>> clusterColorMap = new HashMap<Integer, ArrayList<Integer>>();
	
	static double[][] weights;
	static HashMap<ArrayList<Integer>, ArrayList<Integer>> conversionMapping = new HashMap<ArrayList<Integer>, ArrayList<Integer>>();
	static HashMap<ArrayList<Integer>, ArrayList<Integer>> clusterMapping = new HashMap<ArrayList<Integer>,ArrayList<Integer>>();
	//static HashMap<Integer, ArrayList<Integer>> outBwMap = new HashMap<Integer, ArrayList<Integer>>();
	
	
	public static void main(String[] args) {
		
		Colorization colorization = new Colorization();
		colorization.readData();
		
		System.out.println(trainingStart + ", " + trainingEnd + ", " + testStart + ", " + testEnd + ", " + validationStart + ", " + validationEnd);

		clusterBW();
		clusterColor();

		for(int i=0; i<inBwMap.size(); i++) {
			//conversionMapping.put(convertToBWCenter(inBwMap.get(i)), convertToColorCenter(inColorMap.get(i)));
			
			clusterBwMap.put(i, convertToBWCenter(inBwMap.get(i)));
			clusterColorMap.put(i,  convertToColorCenter(inColorMap.get(i)));
		}
		
		//extractColor();
		
		//neuralNetworkMap();		
//		for (int i=0; i<weights.length; i++) {
//			for (int j=0; j<weights[0].length; j++) {
//				System.out.print(weights[i][j] + " ");
//			}
//			System.out.println();
//		}
		//applyNeuralNetwork();
		
		getMap();
		
		
		extractData();
	}
	
	public void readData() {
	
		String colorCurrentLine = "", bwCurrentLine = "";
		FileReader freader = null, freader1 = null; 
		BufferedReader breader = null, breader1 = null;
		int bwCount = 0, colorCount = 0;
		
		try {
			freader = new FileReader(IN_COLOR_PATH);
			breader = new BufferedReader(freader);
			
			freader1 = new FileReader(IN_BW_PATH);
			breader1 = new BufferedReader(freader1);
			
			while ((colorCurrentLine = breader.readLine()) != null && (bwCurrentLine = breader1.readLine()) != null) {
				
				String[] colorVal = colorCurrentLine.split(",");
				ArrayList<Integer> colorValues = new ArrayList<Integer>();
				for(int i = 0; i < colorVal.length; i++) {
					colorValues.add(Integer.parseInt(colorVal[i].trim()));
				}
				
				String[] bwVal = bwCurrentLine.split(",");
				ArrayList<Integer> bwValues = new ArrayList<Integer>();
				for(int i = 0; i < bwVal.length; i++) {
					bwValues.add(Integer.parseInt(bwVal[i].trim()));
				}
				
				inColorMap.put(colorCount++, colorValues);
				inBwMap.put(bwCount++, bwValues);
				
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				if (breader != null)
					breader.close();
				if (freader != null)
					freader.close();
				if (breader1 != null)
					breader1.close();
				if (freader1 != null)
					freader1.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}
	
	public static void clusterBW() {

		for (int j=0; j<inBwclusters.length; j++) { //initialize clusters
			int rand = (int)(Math.random() * trainingEnd+1);
			
			for (int k=0; k<9; k++) {
				inBwclusters[j][k] = inBwMap.get(rand).get(k);
			}
		}

		//System.out.println("Initial Clusters:");
		//print2dBWArray(inBwclusters);
			
		int[] clusterClassification = new int[TOTAL_LINES]; 
			
		for (int j=0; j<maxReclassification; j++) { //iterations of reclustering
				
			for (int k=0; k<TOTAL_LINES; k++) { //classify each datapoint
					
				double dist = 0;
				
				for (int l=0; l<inBwclusters.length; l++) { //check each cluster
					
					double newDist = 0;
					newDist = calcWeightedBWDist(inBwMap.get(k), inBwclusters[l]);
						
					if (dist == 0 || newDist < dist) {
						dist = newDist;
						clusterClassification[k] = l;
					}
					newDist = 0;
				}	
			}
			
			if (j==maxReclassification-1) {
				bwCluster = clusterClassification;
				break;
			}
				
			//recluster
			int[][] newClusters = new int[numClusters][9];
			int[] totalDataPerCluster = new int[numClusters];
				
			for (int k=0; k<TOTAL_LINES; k++) {
				totalDataPerCluster[clusterClassification[k]] += 1;
					
				for (int l=0; l<9; l++) { //clusters without observations are dropped
					newClusters[clusterClassification[k]][l] += inBwMap.get(k).get(l);
				}
			}
				
			for (int k=0; k<numClusters; k++) {
				for (int l=0; l<9; l++) {
					if (totalDataPerCluster[k] != 0) {
						newClusters[k][l] = (newClusters[k][l]/totalDataPerCluster[k]);
					}
				}
			}
		
			inBwclusters = newClusters;

		}
		
		//System.out.println("Final Clusters: ");
		//print2dBWArray(inBwclusters);
		
		System.out.println();
		
		int zero = 0;
		for (int j=0; j<inBwclusters.length; j++) {
			if (inBwclusters[j][0] == 0 && inBwclusters[j][1] == 0 && inBwclusters[j][2] == 0 && inBwclusters[j][3] == 0 && inBwclusters[j][4] == 0) {
				zero++;
			}
		}
		System.out.println("Removed clusters: " + zero);
		
		//calculate total error
		double error = 0;
		for (int j=0; j<TOTAL_LINES; j++) {
			error = calcWeightedBWDist(inBwMap.get(j), inBwclusters[clusterClassification[j]]);
		}
		System.out.println("Error:" + error);
		
	}
	
	public static void clusterColor() {
			
		for (int j=0; j<inColorclusters.length; j++) { //initialize clusters
			int rand = (int)(Math.random() * trainingEnd+1);
			
			for (int k=0; k<3; k++) {
				inColorclusters[j][k] = inColorMap.get(rand).get(k);
			}
		}
		
		//System.out.println("Initial Clusters:");
		//print2dColorArray(inColorclusters);
			
		int[] clusterClassification = new int[TOTAL_LINES]; 
			
		for (int j=0; j<maxReclassification; j++) { //iterations of reclustering
				
			for (int k=0; k<TOTAL_LINES; k++) { //classify each datapoint
					
				double dist = 0;
				
				for (int l=0; l<inColorclusters.length; l++) { //check each cluster
					
					double newDist = 0;
						
					newDist = calcWeightedColorDist(inColorMap.get(k), inColorclusters[l]);
						
					if (dist == 0 || newDist < dist) {
						dist = newDist;
						clusterClassification[k] = l;
					}
					newDist = 0;
				}	
			}
			
			if (j==maxReclassification-1) {
				colorCluster = clusterClassification;
				break;
			}
				
			//recluster
			int[][] newClusters = new int[numClusters][3];
			int[] totalDataPerCluster = new int[numClusters];
				
			for (int k=0; k<TOTAL_LINES; k++) {
				totalDataPerCluster[clusterClassification[k]] += 1;
					
				for (int l=0; l<3; l++) { //clusters without observations are dropped
					newClusters[clusterClassification[k]][l] += inColorMap.get(k).get(l);
					//newClusters[clusterClassification[k]][l] += Math.pow(inColorMap.get(k).get(l),2);
				}
			}
				
			for (int k=0; k<numClusters; k++) {
				for (int l=0; l<3; l++) {
					if (totalDataPerCluster[k] != 0) {
						//System.out.println(newClusters[k][l]);
						newClusters[k][l] = (newClusters[k][l]/totalDataPerCluster[k]);
						//newClusters[k][l] = (int)Math.sqrt((double)(newClusters[k][l])/totalDataPerCluster[k]);
						//System.out.println(newClusters[k][l]);
					}
				}
			}
		
			inColorclusters = newClusters;
		}

		
		//System.out.println("Final Clusters: ");
		//print2dColorArray(inColorclusters);
		
		System.out.println();
		
		int zero = 0;
		for (int j=0; j<inColorclusters.length; j++) {
			if (inColorclusters[j][0] == 0 && inColorclusters[j][1] == 0 && inColorclusters[j][2] == 0) {
				zero++;
			}
		}
		System.out.println("Removed clusters: " + zero);
		
		//calculate total error
		double error = 0;
		for (int j=0; j<TOTAL_LINES; j++) {
			error = calcWeightedColorDist(inColorMap.get(j), inColorclusters[clusterClassification[j]]);
		}
		System.out.println("Error:" + error);
		
	}

	public static ArrayList<Integer> convertToBWCenter(ArrayList<Integer> input) {
		
		double dist = Integer.MAX_VALUE;
		int pos = 0;
		for(int i=0; i<inBwclusters.length ;i++) {
			
			if (inBwclusters[i][0] == 0 && inBwclusters[i][1] == 0 && inBwclusters[i][2] == 0 && inBwclusters[i][3] == 0 && inBwclusters[i][4] == 0) {
				continue;
			}
			
			double newDist = 0;
			newDist = calcWeightedBWDist(input, inBwclusters[i]);
		
			if(newDist < dist) {
				dist = newDist;
				pos = i;
			}
		}
		
		ArrayList<Integer> output = new ArrayList<Integer>();
		for(int i=0; i<inBwclusters[pos].length; i++) {
			output.add(inBwclusters[pos][i]);
		}
		return output;
	}
	
	public static ArrayList<Integer> convertToColorCenter(ArrayList<Integer> input) {
		
		double dist = Integer.MAX_VALUE;
		int pos = 0;
		for(int i=0; i<inColorclusters.length ;i++) {
			
			if (inColorclusters[i][0] == 0 && inColorclusters[i][1] == 0 && inColorclusters[i][2] == 0) {
				continue;
			}
			
			double newDist = 0;
			for (int m=0; m<3; m++) {
				newDist = Math.abs(input.get(m) - inColorclusters[i][m]);
			}
		
			if(newDist < dist) {
				dist = newDist;
				pos = i;
			}
		}
		
		ArrayList<Integer> output = new ArrayList<Integer>();
		for(int i=0; i<inColorclusters[pos].length; i++) {
			output.add(inColorclusters[pos][i]);
		}
		return output;
	}

	public static void neuralNetworkMap() {
//		int[] g1 = {1, 0, 0, 0, 0, 0, 0, 0, 0};
//		int[] g2 = {0, 1, 0, 0, 0, 0, 0, 0, 0};
//		int[] g3 = {0, 0, 1, 0, 0, 0, 0, 0, 0};
//		int[] g4 = {0, 0, 0, 1, 0, 0, 0, 0, 0};
//		int[] g5 = {0, 0, 0, 0, 1, 0, 0, 0, 0};
//		int[] g6 = {0, 0, 0, 0, 0, 1, 0, 0, 0};
//		int[] g7 = {0, 0, 0, 0, 0, 0, 1, 0, 0};
//		int[] g8 = {0, 0, 0, 0, 0, 0, 0, 1, 0};
//		int[] g9 = {0, 0, 0, 0, 0, 0, 0, 0, 1};
//		
//		int[] g123 = {1, 1, 1, 0, 0, 0, 0, 0, 0};
//		int[] g456 = {0, 0, 0, 1, 1, 1, 0, 0, 0};
//		int[] g789 = {0, 0, 0, 0, 0, 0, 1, 1, 1};
//		
//		int[] g2468 = {0, 1, 0, 1, 0, 1, 0, 1, 0};
//		int[] g1379 = {1, 0, 1, 0, 0, 0, 1, 0, 1};
//		int[] gnot4 = {1, 1, 1, 1, 0, 1, 1, 1, 1};
		
		double rError, gError, bError;
		double learningRate = 0.8;
		
		double rwOuter = 0.40; 
		double rwCenter = 0.30;
		//double rwTop = 0.000125; 
		//double rwmiddleRow = 0.000125;
		//double rwBottom = 0.000125/3; 
		//double rwLeft = 0.000125;
		//double rwmiddleColumn = 0.000125;
		//double rwRight = 0.000125;
		int rConst = 0;
		//double rwPlus = 0.05;
		//double rwCorner = 0.05;
		//double rwTotal = rwOuter + rwCenter + rwTop + rwmiddleRow + rwBottom + rwLeft + rwmiddleColumn + rwRight;// + rwPlus + rwCorner;
		
		double gwOuter = 0.40; 
		double gwCenter = 0.30;
		double gwTop = 0.000125; 
		double gwmiddleRow = 0.000125;
		double gwBottom = 0.000125; 
		double gwLeft = 0.000125;
		double gwmiddleColumn = 0.000125;
		double gwRight = 0.000125;
		int gConst = 0;
		//double gwPlus = 0.05;
		//double gwCorner = 0.05;
		//double gwTotal = gwOuter + gwCenter + gwTop + gwmiddleRow + gwBottom + gwLeft + gwmiddleColumn + gwRight;// + gwPlus + gwCorner;
		
		double bwOuter = 0.40; 
		double bwCenter = 0.30;
		double bwTop = 0.000125; 
		double bwmiddleRow = 0.000125;
		double bwBottom = 0.000125; 
		double bwLeft = 0.000125;
		double bwmiddleColumn = 0.000125;
		double bwRight = 0.000125;
		int bConst = 0;
		//double bwPlus = 0.05;
		//double bwCorner = 0.05;
		//double bwTotal = bwOuter + bwCenter + bwTop + bwmiddleRow + bwBottom + bwLeft + bwmiddleColumn + bwRight;// + bwPlus + bwCorner;
		
		
		
		for (int i=0; i<100; i++) {
			System.out.println(i);
			
			int rand = (int)(Math.random()*testStart);
			System.out.println("random: " + rand);
			ArrayList<Integer> current = clusterBwMap.get(rand);
			System.out.println("center: " + current.get(4));
			System.out.println(current.get(0) + " , " + 
					current.get(1) + " , " + 
					current.get(2) + " , " + 
					current.get(3) + " , " + 
					current.get(4) + " , " + 
					current.get(5) + " , " + 
					current.get(6));// + " , " + 
					//current.get(7) + " , " + 
					//current.get(8));
			
			double outer = (current.get(0) + current.get(1) + current.get(2) + current.get(3) + current.get(5) + current.get(6) + current.get(7) + current.get(8))/8.0;
			double center = current.get(4);
			
			double top = current.get(0) * current.get(1) * current.get(2);
			double middleRow = current.get(3) * current.get(4) * current.get(5);
			double bottom = current.get(6) * current.get(7) * current.get(8);
			
			double left = current.get(0) * current.get(3) * current.get(6);
			double middleColumn = current.get(1) * current.get(4) * current.get(7);
			double right = current.get(2) * current.get(5) * current.get(8);
			
			//double plus = current.get(1) * current.get(3) * current.get(4) * current.get(5) * current.get(7);
			//double corner = current.get(0) * current.get(2) * current.get(6) * current.get(8);
			
			//System.out.println(plus);
			
			rError = (rwOuter*outer + rwCenter*center //+ rwTop*top + rwmiddleRow*middleRow + rwBottom*bottom + 
						//rwLeft*left + rwmiddleColumn*middleColumn + rwRight*right 
					+ rConst) - clusterColorMap.get(rand).get(0);// + rwPlus*plus + rwCorner*corner);
			gError = (gwOuter*outer + gwCenter*center + gwTop*top + gwmiddleRow*middleRow + gwBottom*bottom + 
					gwLeft*left + gwmiddleColumn*middleColumn + gwRight*right + gConst) - clusterColorMap.get(rand).get(1);// + gwPlus*plus + gwCorner*corner);;
			bError = (bwOuter*outer + bwCenter*center + bwTop*top + bwmiddleRow*middleRow + bwBottom*bottom + 
					bwLeft*left + bwmiddleColumn*middleColumn + bwRight*right + bConst) - clusterColorMap.get(rand).get(2);// + bwPlus*plus + bwCorner*corner);
			
			System.out.println(rError + ", " + gError + ", " + bError);
			System.out.println(rwOuter*outer + ", " + rwCenter*center + ", " //+rwTop*top + ", " +rwmiddleRow*middleRow + ", " +rwBottom*bottom + ", " +
						//rwLeft*left + ", " +rwmiddleColumn*middleColumn + ", " +rwRight*right + ", " 
					+ rConst);// + ", " +rwPlus*plus + ", " +rwCorner*corner);
			
			rwOuter = rwOuter - learningRate*rError*2*outer;
			rwCenter = rwCenter - learningRate*rError*2*center;
			//rwTop = rwTop - learningRate*rError*2*top; 
			//rwmiddleRow = rwmiddleRow - learningRate*rError*2*middleRow*(rwmiddleRow/rwTotal);
			//rwBottom = rwBottom - learningRate*rError*2*bottom*(rwBottom/rwTotal); 
			//rwLeft = rwLeft - learningRate*rError*2*left*(rwLeft/rwTotal);
			//rwmiddleColumn = rwmiddleColumn - learningRate*rError*2*middleColumn*(rwmiddleColumn/rwTotal);
			//rwRight = rwRight - learningRate*rError*2*right*(rwRight/rwTotal);
			rConst = (int)(rConst - learningRate*2*rError);
			//rwPlus = learningRate*rError*(rwPlus/rTotal)*plus + rwPlus;
			//rwCorner = learningRate*rError*(rwCorner/rTotal)*corner + rwCorner;
			
			gwOuter = gwOuter - learningRate*gError*2*outer;
			gwCenter = gwCenter - learningRate*gError*2*center;
			gwTop = gwTop - learningRate*gError*2*top; 
			gwmiddleRow = gwmiddleRow - learningRate*gError*2*middleRow;
			gwBottom = gwBottom - learningRate*gError*2*bottom; 
			gwLeft = gwLeft - learningRate*gError*2*left;
			gwmiddleColumn = gwmiddleColumn - learningRate*gError*2*middleColumn;
			gwRight = gwRight - learningRate*gError*2*right;
			gConst = (int)(gConst - learningRate*2*gError);
			//gwPlus = learningRate*gError*(gwPlus/gTotal)*plus + gwPlus;
			//gwCorner = learningRate*gError*(gwCorner/gTotal)*corner + gwCorner;
			
			bwOuter = bwOuter - learningRate*bError*2*outer;
			bwCenter = bwCenter - learningRate*bError*2*center;
			bwTop = bwTop - learningRate*bError*2*top; 
			bwmiddleRow = bwmiddleRow - learningRate*bError*2*middleRow;
			bwBottom = bwBottom - learningRate*bError*2*bottom; 
			bwLeft = bwLeft - learningRate*bError*2*left;
			bwmiddleColumn = bwmiddleColumn - learningRate*bError*2*middleColumn;
			bwRight = bwRight - learningRate*bError*2*right;
			bConst = (int)(bConst - learningRate*2*bError);
			//bwPlus = learningRate*bError*(bwPlus/bTotal)*plus + bwPlus;
			//bwCorner = learningRate*bError*(bwCorner/bTotal)*corner + bwCorner;
			
			
			//rwTotal = rwOuter + rwCenter + rwTop + rwmiddleRow + rwBottom + rwLeft + rwmiddleColumn + rwRight;// + rwPlus + rwCorner;
			//gTotal = gwOuter + gwCenter + gwTop + gwmiddleRow + gwBottom + gwLeft + gwmiddleColumn + gwRight;// + gwPlus + gwCorner;
			//bTotal = bwOuter + bwCenter + bwTop + bwmiddleRow + bwBottom + bwLeft + bwmiddleColumn + bwRight;// + bwPlus + bwCorner;
			
			if (learningRate < 0.001) {
				learningRate = 0.001;
			} else {
				learningRate = learningRate/1.5;
			}
		}
		
		weights = new double[][] {
			{rwOuter, rwCenter, //rwTop, rwmiddleRow, rwBottom, rwLeft, rwmiddleColumn, rwRight, 
				rConst},// rwPlus, rwCorner},
			{gwOuter, gwCenter, gwTop, gwmiddleRow, gwBottom, gwLeft, gwmiddleColumn, gwRight, gConst},// gwPlus, gwCorner},
			{bwOuter, bwCenter, bwTop, bwmiddleRow, bwBottom, bwLeft, bwmiddleColumn, bwRight, bConst}//, bwPlus, bwCorner}
		};
		
	}
	
	public static void getMap() { //maps BWCluster to Color Cluster
		HashMap<ArrayList<Integer>, ArrayList<Integer>> map = new HashMap<ArrayList<Integer>, ArrayList<Integer>>();
		
		for (int i=0; i<clusterBwMap.size(); i++) { //check all inBW clusters
			//System.out.println(i);
			
			ArrayList<Integer> current = clusterBwMap.get(i); //current cluster
			
			if (map.containsKey(current)) continue; //if already counted, continue
			
			HashMap<ArrayList<Integer>, Integer> count = new HashMap<ArrayList<Integer>, Integer>(); //holds count for ColorCluster for current BWCluster
			
			
			for (int j=i; j<clusterBwMap.size(); j++) { 
				ArrayList<Integer> pointer = clusterBwMap.get(j);
				
				if (current.get(0) == pointer.get(0) && 
						current.get(1) == pointer.get(1) && 
						current.get(2) == pointer.get(2) && 
						current.get(3) == pointer.get(3) &&
						current.get(4) == pointer.get(4)) {
					
					ArrayList<Integer> mappedColor = clusterColorMap.get(j);
					if (count.containsKey(mappedColor)) {
						count.put(mappedColor, count.get(mappedColor)+1);
					} else {
						count.put(mappedColor, 1);
					}
				}
				
			}
			
			//get key with largest count
			Map.Entry<ArrayList<Integer>, Integer> maxEntry = null;
			ArrayList<Integer> mappedColor;
			
			
			for (Map.Entry<ArrayList<Integer>, Integer> entry : count.entrySet()) {
				if (maxEntry == null || entry.getValue().compareTo(maxEntry.getValue()) > 0 ){
					maxEntry = entry;
				}
			}
			
			//if (maxEntry != null) {
				mappedColor = maxEntry.getKey();
				map.put(current, mappedColor);
		
			//}
			
			//System.out.println(current.get(0) + " -> " + map.get(current).get(0));

		}
		
		clusterMapping = map;
	}
	
	
	public static void extractData() {
		
		String bwCurrentLine = "";
		FileReader freader = null; 
		BufferedReader breader = null;
		try {
			RandomAccessFile randomFile = new RandomAccessFile(OUT_COLOR_PATH, "rw");
			randomFile.setLength(0);
			
			freader = new FileReader(OUT_BW_PATH);
			breader = new BufferedReader(freader);
			
			while ((bwCurrentLine = breader.readLine()) != null) {
				String[] bwVal = bwCurrentLine.split(",");
				ArrayList<Integer> bwValues = new ArrayList<Integer>();
				for(int i = 0; i < bwVal.length; i++) {
					bwValues.add(Integer.parseInt(bwVal[i].trim()));
				}
				
				ArrayList<Integer> colorValues = clusterMapping.get(convertToBWCenter(bwValues));
				
				
				StringBuffer sb = new StringBuffer();
				for(int i = 0; i < colorValues.size(); i++) {
					if(i < colorValues.size()-1)
						sb.append(colorValues.get(i)+",");
					else
						sb.append(colorValues.get(i));
				}
				    
				 long fileLength = randomFile.length();
				 randomFile.seek(fileLength);
				 randomFile.writeBytes(sb+"\n");
			}
		
			randomFile.close();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				if (breader != null)
					breader.close();
				if (freader != null)
					freader.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}
	
	public static void extractColor() {
		
		String bwCurrentLine = "";
		FileReader freader = null; 
		BufferedReader breader = null;
		try {
			RandomAccessFile randomFile = new RandomAccessFile(OUT_COLOR_PATH, "rw");
			randomFile.setLength(0);
			
			freader = new FileReader(OUT_BW_PATH);
			breader = new BufferedReader(freader);
			
			//while ((bwCurrentLine = breader.readLine()) != null) {
			for (int i=0; i<clusterColorMap.size(); i++) {
				
				ArrayList<Integer> colorValues = new ArrayList<Integer>();

				colorValues = clusterColorMap.get(i);
				
				
				StringBuffer sb = new StringBuffer();
				for(int j = 0; j < colorValues.size(); j++) {
					if(j < colorValues.size()-1)
						sb.append(colorValues.get(j)+",");
					else
						sb.append(colorValues.get(j));
				}
				    
				 long fileLength = randomFile.length();
				 randomFile.seek(fileLength);
				 randomFile.writeBytes(sb+"\n");
			}
		
			randomFile.close();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				if (breader != null)
					breader.close();
				if (freader != null)
					freader.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}
	
	public static void applyNeuralNetwork() {
		
		String bwCurrentLine = "";
		FileReader freader = null; 
		BufferedReader breader = null;
		try {
			RandomAccessFile randomFile = new RandomAccessFile(OUT_COLOR_PATH, "rw");
			randomFile.setLength(0);
			
			freader = new FileReader(OUT_BW_PATH);
			breader = new BufferedReader(freader);
			
			//while ((bwCurrentLine = breader.readLine()) != null) {
			for (int i=0; i<clusterBwMap.size(); i++) {
				
				ArrayList<Integer> colorValues = new ArrayList<Integer>();

				colorValues = getColorsFromNN(clusterBwMap.get(i));
				
				StringBuffer sb = new StringBuffer();
				for(int j = 0; j < colorValues.size(); j++) {
					if(j < colorValues.size()-1)
						sb.append(colorValues.get(j)+",");
					else
						sb.append(colorValues.get(j));
				}
				    
				 long fileLength = randomFile.length();
				 randomFile.seek(fileLength);
				 randomFile.writeBytes(sb+"\n");
			}
		
			randomFile.close();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				if (breader != null)
					breader.close();
				if (freader != null)
					freader.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}
	
	public static ArrayList<Integer> getColorsFromNN(ArrayList<Integer> BWCluster){
		int bwc1 = BWCluster.get(0);
		int bwc2 = BWCluster.get(1);
		int bwc3 = BWCluster.get(2);
		int bwc4 = BWCluster.get(3);
		int bwc5 = BWCluster.get(4);
		int bwc6 = BWCluster.get(5);
		int bwc7 = BWCluster.get(6);
		int bwc8 = BWCluster.get(7);
		int bwc9 = BWCluster.get(8);
		
		int red = (int)(weights[0][0]*(bwc1+bwc2+bwc3+bwc4+bwc6+bwc7+bwc8+bwc9) +
				weights[0][1]*bwc5 +
				weights[0][2]*(bwc1*bwc2*bwc3) + 
				weights[0][3]*(bwc4*bwc5*bwc6) +
				weights[0][4]*(bwc7*bwc8*bwc9) +
				weights[0][5]*(bwc1*bwc4*bwc7) +
				weights[0][6]*(bwc2*bwc5*bwc8) + 
				weights[0][7]*(bwc3*bwc6*bwc9) +
				weights[0][8]);// + 
				//weights[0][8]*(bwc2*bwc4*bwc5*bwc6*bwc8) + 
				//weights[0][9]*(bwc1*bwc3*bwc7*bwc9));
		
		int green = (int)(weights[1][0]*(bwc1+bwc2+bwc3+bwc4+bwc6+bwc7+bwc8+bwc9) +
				weights[1][1]*bwc5 +
				weights[1][2]*(bwc1*bwc2*bwc3) + 
				weights[1][3]*(bwc4*bwc5*bwc6) +
				weights[1][4]*(bwc7*bwc8*bwc9) +
				weights[1][5]*(bwc1*bwc4*bwc7) +
				weights[1][6]*(bwc2*bwc5*bwc8) + 
				weights[1][7]*(bwc3*bwc6*bwc9) +
				weights[1][8]);// + 
				//weights[1][8]*(bwc2*bwc4*bwc5*bwc6*bwc8) + 
				//weights[1][9]*(bwc1*bwc3*bwc7*bwc9));
		
		int blue = (int)(weights[2][0]*(bwc1+bwc2+bwc3+bwc4+bwc6+bwc7+bwc8+bwc9) +
				weights[2][1]*bwc5 +
				weights[2][2]*(bwc1*bwc2*bwc3) + 
				weights[2][3]*(bwc4*bwc5*bwc6) +
				weights[2][4]*(bwc7*bwc8*bwc9) +
				weights[2][5]*(bwc1*bwc4*bwc7) +
				weights[2][6]*(bwc2*bwc5*bwc8) + 
				weights[2][7]*(bwc3*bwc6*bwc9) + 
				weights[2][8]);// + 
				//weights[2][8]*(bwc2*bwc4*bwc5*bwc6*bwc8) + 
				//weights[2][9]*(bwc1*bwc3*bwc7*bwc9));
				
		ArrayList<Integer> colors = new ArrayList<Integer>();
		colors.add(red);
		colors.add(green);
		colors.add(blue);
		
		return colors;
	}
	
	public static double calcDist(ArrayList<Integer> color1, ArrayList<Integer> color2) {
		Double dist = Math.sqrt(( 2 * Math.pow(color1.get(0) - color2.get(0), 2) + 4 * Math.pow(color1.get(1) - color2.get(1), 2) + 3 * Math.pow(color1.get(2) - color2.get(2), 2) ));
		return dist.intValue();
	}
	
	public static double calcWeightedBWDist(ArrayList<Integer> arr1, int[] arr2) {
		double pixel1 = 0.025*Math.pow(arr1.get(0) - arr2[0], 2);
		double pixel2 = 0.025*Math.pow(arr1.get(1) - arr2[1], 2);
		double pixel3 = 0.025*Math.pow(arr1.get(2) - arr2[2], 2);
		double pixel4 = 0.025*Math.pow(arr1.get(3) - arr2[3], 2);
		double pixel5 = 0.8 * Math.pow(arr1.get(4) - arr2[4], 2);
		double pixel6 = 0.025*Math.pow(arr1.get(5) - arr2[5], 2);
		double pixel7 = 0.025*Math.pow(arr1.get(6) - arr2[6], 2);
		double pixel8 = 0.025*Math.pow(arr1.get(7) - arr2[7], 2);
		double pixel9 = 0.025*Math.pow(arr1.get(8) - arr2[8], 2);
		
		double distance = Math.pow(pixel1+pixel2+pixel3+pixel4+pixel5+pixel6+pixel7+pixel8+pixel9, 0.5);
		return distance;
	}
	
	public static double calcWeightedColorDist(ArrayList<Integer> arr1, int[] arr2) {
		double red = 2*Math.pow(arr1.get(0) - arr2[0], 2);
		double green = 4*Math.pow(arr1.get(1) - arr2[1], 2);
		double blue = 3*Math.pow(arr1.get(2) - arr2[2], 2);
		
		double distance = Math.sqrt(red+green+blue);
		return distance;
	}
	
	public static void print2dBWArray(int[][] array) {
		for (int i=0; i<array.length; i++) {
			System.out.println("Cluster: " + (i+1));
			System.out.println(array[i][0] + " " + array[i][1] + " " + array[i][2]);
			System.out.println(array[i][3] + " " + array[i][4] + " " + array[i][5]);
			System.out.println(array[i][6] + " " + array[i][7] + " " + array[i][8]);
			System.out.println();
		}
	}
	
	public static void print2dColorArray(int[][] array) {
		for (int i=0; i<array.length; i++) {
			System.out.println("Cluster: " + (i+1));
			System.out.println(array[i][0] + " " + array[i][1] + " " + array[i][2]);
			System.out.println();
		}
	}
}
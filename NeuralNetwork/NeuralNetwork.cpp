#include "trainingData.h"
#include "neuron.h"
#include "net.h"
#include <iostream>
#include <fstream>

//fix this crap
template <class T>
void showVectorVals(string label, vector<T> &v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}

void fillin(const string filename)
{
	fstream out(filename, fstream::out);
	out << "topology: 2 1 1" << std::endl;
	for (int i = 0; i < 20000; ++i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
		int t = !(n1 && n2);
		out << "in: " << n1 << ".0 " << n2 << ".0" << std::endl;
		out << "out: " << t << ".0 " << std::endl;
	}
}

int main()
{
	fillin("test.txt");
	TrainingData trainData("test.txt");
	//e.g., {3, 2, 1 }
	vector<unsigned> topology;
	/*topology.push_back(2);
	topology.push_back(4);
	topology.push_back(1);*/

	trainData.getTopology(topology);
	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	fstream error;
	error.open("error.txt", fstream::out);

	while (!trainData.isEof())
	{
		++trainingPass;
		//cout << endl << "Pass" << trainingPass;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) break;
		//showVectorVals(": Inputs :", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual results:
		myNet.getResults(resultVals);
		//showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		//showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how well the training is working, average over recent
		//cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
		error << trainingPass << "\t" << myNet.getRecentAverageError() << std::endl;
	}
	error.close();

	vector<double> userTry;
	while (true)
	{
		try 
		{
			std::cout << "clear" << std::endl;
			userTry.clear();
			std::cout << "Type first input" << std::endl;
			int num;
			cin >> num;
			if (num != 0 && num != 1) throw num;
			userTry.push_back(num);
			std::cout << "You typed : " << userTry[0] << std::endl;
			std::cout << "Type second input" << std::endl;
			cin >> num;
			if (num != 0 && num != 1) throw num;
			userTry.push_back(num);
			std::cout << "You typed : " << userTry[1] << std::endl;
			myNet.feedForward(userTry);
			vector<double> results;
			myNet.getResults(results);
			showVectorVals("result = ",results);
		}
		catch (int &a)
		{
			std::cout << "You typed wrong number " << a << std::endl;
		}
	}
	cout << endl << "Done" << endl;
	system("PAUSE");
}
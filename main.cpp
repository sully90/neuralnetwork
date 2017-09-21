// neural-net.cpp - implicity, and fully, connected neural network. Based on the example by David Miller.
// g++ neural-net.cpp -o neural-net

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>

#include "Neuron.h"
#include "PredictionData.h"

#define PI 3.14159265

using namespace std;

// Define utility functions
template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

std::vector<double> split_to_double(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));

    vector<double> res;
    for (int i = 0; i < elems.size(); i++) {
        res.push_back(atof(elems[i].c_str()));
    }
    return res;
}

bool isInside(const std::string & str, char c)
{
    return str.find(c) != std::string::npos;
}

// *************************** class Net *************************** 

class Net
{
public:
    Net(const vector<unsigned> &topology);
    vector<unsigned> getTopology(void) const;
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }
    void exportToCSV(const string csvFilename);
    static Net loadCSV(const string csvFilename);
    static double error_thresh;

private:
    vector<Layer> m_layers;  // m_layers[layerNum][neuronNum]
    vector<unsigned> m_topology;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor = 100.0;
    void importFromCSV(ifstream &csvFile);
};

double Net::error_thresh = 0.2;

void Net::importFromCSV(ifstream &csvFile)
{
    char delim = ',';
    string line;
    unsigned numLayers = m_topology.size();

    for (unsigned layerNum = 0; layerNum < numLayers - 1; layerNum++) {
        Layer &currentLayer = m_layers[layerNum];

        for (unsigned neuronNum = 0; neuronNum < currentLayer.size(); neuronNum++) {
            Neuron &currentNeuron = currentLayer[neuronNum];

            // Read a line
            getline(csvFile, line);

            vector<double> weights;

            if (isInside(line, delim)) {
                weights = split_to_double(line, delim);
            } else {
                weights.push_back(atof(line.c_str()));
            }
            currentNeuron.importConnectionWeights(weights);
        }
    }
}


Net Net::loadCSV(const string csvFilename)
{
    vector<unsigned> topology;
    ifstream csvFile;
    csvFile.open(csvFilename);

    string line;
    string label;

    getline(csvFile, line);
    stringstream ss(line);
    ss >> label;
    if (csvFile.eof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    Net myNet(topology);
    myNet.importFromCSV(csvFile);
    csvFile.close();

    return myNet;
}


void Net::exportToCSV(const string csvFilename)
{
    ofstream csvFile;
    csvFile.open(csvFilename);

    unsigned numLayers = m_topology.size();

    csvFile << "topology: ";
    for (int i = 0; i < m_topology.size(); i++) {
        csvFile << m_topology[i];
        if (i < m_topology.size() - 1) csvFile << " ";
    }
    csvFile << endl;

    for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
        Layer &currentLayer = m_layers[layerNum];

        for (unsigned neuronNum = 0; neuronNum < currentLayer.size(); neuronNum++) {
            Neuron &currentNeuron = currentLayer[neuronNum];
            currentNeuron.exportConnectionWeights(csvFile);
        }
    }

    csvFile.close();
}


vector<unsigned> Net::getTopology(void) const
{
    return m_topology;
}


void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; n++) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}


void Net::backProp(const vector<double> &targetVals)
{
    // Calculate overall net error (RMS)

    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;  // get average error squared
    m_error = sqrt(m_error);  // RMS

    // Implement a recent average measurement
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradiends

    for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--) {  // start with right most hidden layer and count down
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); n++) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--) {  // go through all layers, starting at right most and dont include the input layer (as theres no input weights)
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; n++) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}


void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);  // -1 to account for bias

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); i++) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {  // start with first hidden layer (skip input)
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}


Net::Net(const vector<unsigned> &topology)
{
    m_topology = topology;
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // We have a new layer, now fill it with neurons, and add bias neuron
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
        }

        // Set bias neurons output to 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

void writeVectorVals(ofstream &fstream, vector<double> &inputVals, vector<double> &resultVals)
{
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        fstream << inputVals[i] << " ";
    }

    // Do not change the line within this loop, as it rescales fb from Neural Net units to
    // the correct baryon fraction in units of the cosmic mean.
    for (unsigned i = 0; i < resultVals.size(); ++i) {
        fstream << (((resultVals[i]/2.) + 0.5) * 0.75)/0.16544117647058823 << " ";
    }
    fstream << endl;
}

double RandomFloat(double min, double max)
{
    double r = (double)rand() / (double)RAND_MAX;
    return min + r * (max - min);
}


Net trainNeuralNetwork(int niter, const string trainingFile, bool printStats)
{
    PredictionData trainData(trainingFile);

    vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    int n_neurons = 0;
    for (unsigned i = 0; i < topology.size(); i++) {
        n_neurons += topology[i];
    }

    ostringstream stringStream;
    stringStream << "./errors_" << to_string(n_neurons) << ".txt";

    bool writeErrors = true;

    ofstream errorfile;
    if (writeErrors) {
        string errorFilename = stringStream.str();
        errorfile.open(errorFilename);
    }

    int learning_adjust_count = 0;

    for (unsigned i = 1; i < niter+1; i++) {
        cout << i << endl;
        while (!trainData.isEof()) {
            ++trainingPass;
            if (printStats) cout << endl << "Pass " << trainingPass;

            // Get new input data and feed it forward:
            if (trainData.getNextInputs(inputVals) != topology[0]) {
                break;
            }
            if (printStats) showVectorVals(": Inputs:", inputVals);
            myNet.feedForward(inputVals);

            // Collect the net's actual output results:
            myNet.getResults(resultVals);
            if (printStats) showVectorVals("Outputs:", resultVals);

            // Train the net what the outputs should have been:
            trainData.getTargetOutputs(targetVals);
            if (printStats) showVectorVals("Targets:", targetVals);
            assert(targetVals.size() == topology.back());

            // Backwards propagation loop
            myNet.backProp(targetVals);

            // Modify the learning rate as we become more accurate:
            if (i > 2) {
                if (myNet.getRecentAverageError() <= Net::error_thresh) learning_adjust_count++;
                else learning_adjust_count = 0;
            }

            // Implement a crude but simple adaptive learning rate:
            if (learning_adjust_count >= 1000000) {
                cout << "Adjusting learning rate..." << endl;
                Neuron::eta = std::max(Neuron::eta/2, 0.005);
                Neuron::alpha /= 2;
                Net::error_thresh /= 1.25;
                cout << Neuron::eta << " " << Neuron::alpha << " " << Net::error_thresh << endl;
                learning_adjust_count = 0;
            }

            // Report how well the training is working, average over recent samples:
            if (printStats) {
                cout << "Net recent average error: "
                    << myNet.getRecentAverageError() << endl;
            }

            // Write out the network errors every 50000 passes
            if (writeErrors && trainingPass % 50000 == 0) errorfile << trainingPass << " " << myNet.getRecentAverageError() << endl;
        }
        trainData.rewind();
        trainData.getTopology(topology);
    }

    if(writeErrors) errorfile.close();

    return myNet;
}


string getInputParameter(string param, int argc, char ** argv, bool requirePresent)
{
    for (int i = 1; i < argc; i++) {
        if (i + 1 != argc) {
            if (string(argv[i]) == param) {
                return argv[i + 1];
            }
        }
    }
    if (requirePresent) {
        cout << "Could not find input parameter: " << param;
        exit(0);
    }
    return string();
}


bool inputFlagPresent(string param, int argc, char ** argv)
{
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == param) return true;
    }
    return false;
}


Net initialseFromWeights()
{
    // This function loads the connection weights provided which were used in Sullivan et al. 2017
    string weightsFileName = "weights/NN50.weights";
    Net myNet = Net::loadCSV(weightsFileName);
    return myNet;
}


int main(int argc, char ** argv)
{
    // Modified main to initialse from exported weights and make prediction.
    Net myNet = initialseFromWeights();

    // File containing data for which we want to make predictions
    string predictionFile = getInputParameter("-f", argc, argv, true);

    // If a prediction file was given, feedforward the data
    if (!predictionFile.empty()) {
        cout << predictionFile << endl;
        // The file to which the results will be written
        string outputFile = getInputParameter("-o", argc, argv, true);

        cout << "Writing results to file: " << outputFile << endl;

        ofstream outfile;
        outfile.open(outputFile);

        // Load the prediction data
        PredictionData inputData(predictionFile);

        vector<double> inputVals, resultVals;
        int trainingPass = 0;

        // Feedforward loop
        while (!inputData.isEof()) {
            ++trainingPass;

            // Get new input data and feed it forward:
            if (inputData.getNextInputs(inputVals) != myNet.getTopology()[0]) {
                if (inputVals.size() > 0) {
                    cout << trainingPass << endl;
                    throw invalid_argument("Prediction data does not match topology!");
                }
                else {
                    break;
                }
            }
            myNet.feedForward(inputVals);

            // Collect the net's actual output results:
            myNet.getResults(resultVals);
            writeVectorVals(outfile, inputVals, resultVals);
        }
        outfile.close();
    }
}

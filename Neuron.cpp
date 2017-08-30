// Neuron.cpp
#include "Neuron.h"

using namespace std;

double Neuron::eta = 0.15;
// double Neuron::eta = 0.2;
double Neuron::alpha = 0.25;  // momentum, multiplier of last deltaWeight

void Neuron::importConnectionWeights(vector<double> connectionWeights)
{
    for (int weightNum = 0; weightNum < connectionWeights.size(); weightNum++) {
        m_outputWeights[weightNum].weight = connectionWeights[weightNum];
    }
}

void Neuron::exportConnectionWeights(ofstream &csvFile)
{
    for (int weightNum = 0; weightNum < m_outputWeights.size(); weightNum++) {
        csvFile << m_outputWeights[weightNum].weight;
        if (weightNum < m_outputWeights.size() - 1) csvFile << ",";
    }
    csvFile << endl;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); n++) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        // Individual input, magnified by the gradient and train rate:
        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient
            // Also add momentum = a fraction of the prev. delta weight
            + alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed

    for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    // tanh - output range [-1.0, 1.0]
    // output values must always be in this range
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer

    for (unsigned n = 0; n < prevLayer.size(); n++) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    // If using a function whos derivative is not a function of itself (i.e not dtanh/dx = 1 - tanh(x)^2)
    // then just set m_outputVal to sum and implement the transfer function calculated in the getter
    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; c++) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

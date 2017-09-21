// PREDICTION.h
#ifndef PREDICTION_DATA_H
#define PREDICTION_DATA_H

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

class PredictionData
{
public:
    PredictionData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void close(void) { m_trainingDataFile.close(); }
    void getTopology(vector<unsigned> &topology);
    void rewind(void);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

#endif
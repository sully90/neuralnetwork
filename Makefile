# Declaration of variables
CC = g++
CC_FLAGS = -w -O2
 
# File names
EXEC = neural-net
SOURCES = $(wildcard *.cpp)
OBJECTS = Neuron.o PredictionData.o main.o
 
# Main target
$(EXEC): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXEC)
 
# To obtain object files
Neuron.o: Neuron.cpp Neuron.h
	$(CC) $(CC_FLAGS) -c Neuron.cpp

TrainingData.o: TrainingData.cpp TrainingData.h
	$(CC) $(CC_FLAGS) -c TrainingData.cpp

main: main.o
	$(CC) $(CC_FLAGS) main.o -o $(EXEC)

main.o: main.cpp
	$(CC) $(CC_FLAGS) -c main.cpp	
 
# To remove generated files
clean:
	rm -f $(EXEC) $(OBJECTS)

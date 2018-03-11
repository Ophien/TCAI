/* This source code provides examples on how to use the improved
 * Adaptive Resonance Associative Map.
 * 
 * Copyright(C) 2018 Alysson Ribeiro da Silva
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.If not, see<https://www.gnu.org/licenses/>.
 * 
 * If you want to ask me any question, you can send an e-mail to 
 * Alysson.Ribeiro.Silva@gmail.com entitled as "C# ANN TEST BED"
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TCAI
{
    class TestBed
    {
        public void printLicense()
        {
            Console.WriteLine("This source code provides examples on how to use the improved");
            Console.WriteLine("Adaptive Resonance Associative Map.");
            Console.WriteLine("");
            Console.WriteLine("Copyright(C) 2018 Alysson Ribeiro da Silva");
            Console.WriteLine("");
            Console.WriteLine("This program is free software: you can redistribute it and/or modify");
            Console.WriteLine("it under the terms of the GNU General Public License as published by");
            Console.WriteLine("the Free Software Foundation version 3.");
            Console.WriteLine("");
            Console.WriteLine("This program is distributed in the hope that it will be useful,");
            Console.WriteLine("but WITHOUT ANY WARRANTY; without even the implied warranty of");
            Console.WriteLine("MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the");
            Console.WriteLine("GNU General Public License for more details.");
            Console.WriteLine("");
            Console.WriteLine("You should have received a copy of the GNU General Public License");
            Console.WriteLine("along with this program.If not, see<https://www.gnu.org/licenses/>.");
            Console.WriteLine("");
            Console.WriteLine("If you want to ask me any question, you can send an e-mail to");
            Console.WriteLine("Alysson.Ribeiro.Silva@gmail.com entitled as \"C# ADAPTIVE NEURAL NETWORK\"\n\n\n");
        }

        public void example1_oneField()
        {
            Console.WriteLine("***************************************************************");
            Console.WriteLine("******* TEST 1 - THE ONE FIELD ANN ****************************");
            Console.WriteLine("***************************************************************\n");

            // ---------------------------------------------------------------------
            // ------------ Creating a configuration file --------------------------
            // ---------------------------------------------------------------------
            NetDescription description = new NetDescription();

            // ---------------------------------------------------------------------
            // ------------ Configuring an one field Adaptive Neural Network -------
            // ---------------------------------------------------------------------
            description.activeFields = new bool[] { true }; // defines which fields are active to perform the categorization
            description.adaptiveVigilanceRaising = new bool[] { true }; // defines which fields will be affected by the adaptive vigilance
            description.fuzzyReadout = false; // defines if the prediction will be performed by the fuzzy ARTI readout

            description.featuresSizes = new int[] { 4 }; // defines the fields configuration, their size and quantity
            description.alphas = new double[] { 0.1 }; // defines each field alpha
            description.gammas = new double[] { 0.5 }; // defines each field gamma
            description.learningRate = new double[] { 1.0 }; // defines the learning rate for each field
            description.learningVigilances = new double[] { 1.0 }; // defines the vigilance used for learning for each field
            description.performingVigilances = new double[] { 0.0 }; // defines the vigilance used to perform predictions for each field
            description.adaptiveVigilanceRate = 0.001; // defines how much the vigilance will be added when predicting

            description.fieldsClass = new int[] { FieldTypes.STATE }; // defines each field type
            description.learningOp = new int[] { NeuronLearning.ART_I }; // defines which composite operation will be used by the network when learning
            description.temperatureOp = new int[] { NeuronActivation.ART_I }; // defines which composite operation will be used by the network when performing

            // ---------------------------------------------------------------------
            // ---------- Creating the Adaptive Neural Network ---------------------
            // ---------------------------------------------------------------------
            AdaptiveNeuralNetwork network = new AdaptiveNeuralNetwork(description); // creates the network

            network.printNetStructure();
            network.printNetworkParameters();
            network.setDebug(true);

            // ---------------------------------------------------------------------
            // --- Performing learning operations to learn the observed activity ---
            // ---------------------------------------------------------------------

            // Received external stimulus, what the agent is seeing
            double[] externalStimulus = new double[4];
            externalStimulus[0] = 0.3;
            externalStimulus[1] = 0.3;
            externalStimulus[2] = 0.7;
            externalStimulus[3] = 0.1;

            // The field that will receive the observed stimulus
            int fieldToWrite = 0;

            // Inserting the observation into the network's activity vectors
            network.setInputField(fieldToWrite, externalStimulus);

            // Variable that tells to the network if it needs to learn the observed stimulus
            bool learn = true;

            // Performing a learning operation
            network.prediction(learn);

            // Received external stimulus, what the agent is seeing
            externalStimulus[0] = 0.1;
            externalStimulus[1] = 0.3;
            externalStimulus[2] = 0.5;
            externalStimulus[3] = 0.6;

            // Inserting the observation into the network's activity vectors
            network.setInputField(fieldToWrite, externalStimulus);

            // Variable that tells to the network if it needs to learn the observed stimulus
            learn = true;

            // Performing a learning operation
            network.prediction(learn);

            // ---------------------------------------------------------------------
            // --- Performing a prediction operation to read a neuron cluster ------
            // ---------------------------------------------------------------------

            // Received external stimulus, what the agent is seeing
            externalStimulus[0] = 0.1;
            externalStimulus[1] = 0.3;
            externalStimulus[2] = 0.4;
            externalStimulus[3] = 0.6;

            // The field that will receive the observed stimulus
            fieldToWrite = 0;

            // Inserting the observation into the network's activity vectors
            network.setInputField(fieldToWrite, externalStimulus);

            // Variable that tells to the network if it needs to learn the observed stimulus
            learn = false;

            // Performing a learning operation
            network.prediction(learn);

            // ---------------------------------------------------------------------
            // --- Reading a prediction for the field 0 ----------------------------
            // ---------------------------------------------------------------------

            // The field in which the read operation will be performed
            int fieldToRead = 0;

            // The array that will receive the prediction
            double[] prediction = new double[4];

            // Performing the reading operation
            prediction = network.readPrediction(fieldToRead);

            // ---------------------------------------------------------------------
            // --- Printing the observed prediction --------------------------------
            // ---------------------------------------------------------------------

            // Pause
            Console.Write("Press ENTER to continue...");
            Console.ReadLine();
            Console.WriteLine();
        }

        public void example2_threeFields()
        {
            Console.WriteLine("***************************************************************");
            Console.WriteLine("******* TEST 2 - THE THREE FIELDS ANN (FALCON) ****************");
            Console.WriteLine("***************************************************************\n");

            // ---------------------------------------------------------------------
            // ------------ Creating a configuration file --------------------------
            // ---------------------------------------------------------------------
            NetDescription description = new NetDescription();

            // ---------------------------------------------------------------------
            // ------------ Configuring an one field Adaptive Neural Network -------
            // ---------------------------------------------------------------------
            description.activeFields = new bool[] { true, true, false }; // defines which fields are active to perform the categorization
            description.adaptiveVigilanceRaising = new bool[] { true, false, false }; // defines which fields will be affected by the adaptive vigilance
            description.fuzzyReadout = true; // defines if the prediction will be performed by the fuzzy ARTI readout

            description.featuresSizes = new int[] { 4, 4, 2 }; // defines the fields configuration, their size and quantity
            description.alphas = new double[] { 0.1, 0.1, 0.1 }; // defines each field alpha
            description.gammas = new double[] { 0.5, 0.5, 0.0 }; // defines each field gamma
            description.learningRate = new double[] { 1.0, 1.0, 1.0 }; // defines the learning rate for each field
            description.learningVigilances = new double[] { 1.0, 1.0, 1.0 }; // defines the vigilance used for learning for each field
            description.performingVigilances = new double[] { 0.0, 0.0, 0.0 }; // defines the vigilance used to perform predictions for each field
            description.adaptiveVigilanceRate = 0.001; // defines how much the vigilance will be added when predicting

            description.fieldsClass = new int[] { FieldTypes.STATE, FieldTypes.ACTION, FieldTypes.REWARD }; // defines each field type
            description.learningOp = new int[] { NeuronLearning.ART_I, NeuronLearning.ART_I, NeuronLearning.ART_II }; // defines which composite operation will be used by the network when learning
            description.temperatureOp = new int[] { NeuronActivation.ART_I, NeuronActivation.ART_I, NeuronActivation.ART_II }; // defines which composite operation will be used by the network when performing

            // ---------------------------------------------------------------------
            // ---------- Creating the Adaptive Neural Network ---------------------
            // ---------------------------------------------------------------------
            AdaptiveNeuralNetwork network = new AdaptiveNeuralNetwork(description); // creates the network

            network.printNetStructure();
            network.printNetworkParameters();
            network.setDebug(true);

            // ---------------------------------------------------------------------
            // --- Performing learning operations to learn the observed activity ---
            // ---------------------------------------------------------------------

            // Received external stimulus, what the agent is seeing
            double[] externalStimulusA = new double[2];
            double[] externalStimulusB = new double[2];
            double[] externalStimulusC = new double[1];

            // Define half of the input stimulus, since complement coding will be used for this test
            externalStimulusA[0] = 0.3;
            externalStimulusA[1] = 0.2;

            // Define half of the input stimulus, since complement coding will be used for this test
            externalStimulusB[0] = 0.1;
            externalStimulusB[1] = 0.9;

            // Define half of the input stimulus, since complement coding will be used for this test
            externalStimulusC[0] = 1.0;

            // The field that will receive the observed stimulus
            int fieldToWriteA = 0;
            int fieldToWriteB = 1;
            int fieldToWriteC = 2;

            // Inserting the observation into the network's activity vectors
            network.setInputField(fieldToWriteA, externalStimulusA, ComplementCodingType.MIRRORED);
            network.setInputField(fieldToWriteB, externalStimulusB, ComplementCodingType.MIRRORED);
            network.setInputField(fieldToWriteC, externalStimulusC, ComplementCodingType.MIRRORED);

            // Variable that tells to the network if it needs to learn the observed stimulus
            bool learn = true;

            // Performing a learning operation
            network.prediction(learn);

            // Define half of the input stimulus, since complement coding will be used for this test
            externalStimulusA[0] = 0.1;
            externalStimulusA[1] = 0.6;

            // Define half of the input stimulus, since complement coding will be used for this test
            externalStimulusB[0] = 0.2;
            externalStimulusB[1] = 0.1;

            // Define half of the input stimulus, since complement coding will be used for this test
            externalStimulusC[0] = 0.45;

            // The field that will receive the observed stimulus
            fieldToWriteA = 0;
            fieldToWriteB = 1;
            fieldToWriteC = 2;

            // Inserting the observation into the network's activity vectors
            network.setInputField(fieldToWriteA, externalStimulusA, ComplementCodingType.MIRRORED);
            network.setInputField(fieldToWriteB, externalStimulusB, ComplementCodingType.MIRRORED);
            network.setInputField(fieldToWriteC, externalStimulusC, ComplementCodingType.MIRRORED);

            // Variable that tells to the network if it needs to learn the observed stimulus
            learn = true;

            // Performing a learning operation
            network.prediction(learn);

            // ---------------------------------------------------------------------
            // --- Performing a prediction operation to read a neuron cluster ------
            // ---------------------------------------------------------------------

            // Define half of the input stimulus, since complement coding will be used for this test
            externalStimulusA[0] = 0.3;
            externalStimulusA[1] = 0.2;

            // Define half of the input stimulus, since complement coding will be used for this test
            externalStimulusB[0] = 1.0;
            externalStimulusB[1] = 1.0;

            // Define half of the input stimulus, since complement coding will be used for this test
            externalStimulusC[0] = 1.0;

            // The field that will receive the observed stimulus
            fieldToWriteA = 0;
            fieldToWriteB = 1;
            fieldToWriteC = 2;

            // Inserting the observation into the network's activity vectors
            network.setInputField(fieldToWriteA, externalStimulusA, ComplementCodingType.MIRRORED);
            network.setInputField(fieldToWriteB, externalStimulusB, ComplementCodingType.DIRECT_ACCESS);
            network.setInputField(fieldToWriteC, externalStimulusC, ComplementCodingType.DIRECT_ACCESS);

            // Variable that tells to the network if it needs to learn the observed stimulus
            learn = false;

            // Performing a learning operation
            network.prediction(learn);

            // ---------------------------------------------------------------------
            // --- Reading a prediction for the field 1 ----------------------------
            // ---------------------------------------------------------------------

            // The field in which the read operation will be performed for the ACTION field 1
            int fieldToRead = 1;

            // The array that will receive the prediction
            double[] prediction = new double[4];

            // Performing the reading operation
            prediction = network.readPrediction(fieldToRead);

            // ---------------------------------------------------------------------
            // --- Printing the observed prediction --------------------------------
            // ---------------------------------------------------------------------

            // Pause
            Console.Write("Press ENTER to continue...");
            Console.ReadLine();
            Console.WriteLine();
        }

        static void Main(string[] args)
        {
            TestBed testBed = new TestBed();
            testBed.printLisence();

            testBed.example1_oneField();
            testBed.example2_threeFields();
        }
    }
}

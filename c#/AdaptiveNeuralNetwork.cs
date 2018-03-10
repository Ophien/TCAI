/* This source code provides an improved implementation of the 
 * Adaptive Resonance Associative Map, proposed by Dr. Ah-Hwee Tan
 * as ART Map extension that displays a faster and more stable 
 * behavior for neuron categorization and prediction.
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
 * Alysson.Ribeiro.Silva@gmail.com entitled as "C# ADAPTIVE NEURAL NETWORK"
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TCAI
{
    /// <summary>
    ///  Created by Alysson Ribeiro da Silva, the NetDescription class is responsible for holding the structural design of the Adaptive Neural Network.
    /// </summary>
    class NetDescription
    {
        public int[] featuresSizes;
        public bool[] adaptiveVigilanceRaising;
        public bool fuzzyReadout;
        public bool[] activeFields;
        public int[] temperatureOp;
        public int[] learningOp;
        public int[] fieldsClass;
        public double[] learningRate;
        public double[] learningVigilances;
        public double[] performingVigilances;
        public double[] gammas;
        public double[] alphas;
        public double adaptiveVigilanceRate;
    }

    /// <summary>
    ///  Created by Alysson Ribeiro da Silva, the NeuronTemperatureTuple class is responsible for holding the neuron temperature and its index to facilitate the network's
    ///  internal operations.
    /// </summary>
    class NeuronTemperatureTuple
    {
        // -------------------------------------------------------------------------------------------------------
        public NeuronTemperatureTuple(int neuronIndex, double t)
        {
            this.t = t;
            this.neuronIndex = neuronIndex;
        }
        // -------------------------------------------------------------------------------------------------------
        public double t;
        public int neuronIndex;
        // -------------------------------------------------------------------------------------------------------
    }

    /// <summary>
    ///  Created by Alysson Ribeiro da Silva, the AdaptiveNeuralNetwork class is implements the Adaptive Resonance Associative Map with the composite operations.
    ///  The deployed model enables to use fuzzy ART I, fuzzy ART II, and a proximity metric for neuron cluster categorization.
    ///  This class helps deploying Q-Learning models and Reactive models, based on action masks, to control agents in real-time.
    ///  Moreover, it also posses a perfect miss-match mechanism, and neuron cluster based operations to facilitate its working and to avoid errors when predicting information.
    /// </summary>
    class AdaptiveNeuralNetwork
    {
        // -------------------------------------------------------------------------------------------------------
        /// <summary>
        /// Creates a new ANN with the specified configuration class
        /// </summary>
        /// <param name="config"></param>
        public AdaptiveNeuralNetwork(NetDescription config)
        {
            this.totalFields = config.featuresSizes.Length;
            this.vigilancesRaising = config.adaptiveVigilanceRaising;
            this.fuzzyReadout = config.fuzzyReadout;
            this.activeFields = config.activeFields;
            this.featuresSizes = config.featuresSizes;
            this.temperatureOp = config.temperatureOp;
            this.learningOp = config.learningOp;
            this.fieldsClass = config.fieldsClass;
            this.learningRate = config.learningRate;
            this.learnVigilances = config.learningVigilances;
            this.performVigilances = config.performingVigilances;
            this.gammas = config.gammas;
            this.alphas = config.alphas;
            this.adaptiveVigilanceRate = config.adaptiveVigilanceRate;

            activity = new double[totalFields][];
            for (int field = 0; field < totalFields; field++)
                activity[field] = new double[featuresSizes[field]];

            activitySum = new double[totalFields];

            createNeuron();
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Configures the Q-Learning dynamics parameters 
        /// </summary>
        /// <param name="discount"></param>
        /// <param name="learningRate"></param>
        public void configureQLearningHelper(double discount, double learningRate)
        {
            this.qDiscountParameter = discount;
            this.qLearningRate = learningRate;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Configures the reactive model dynamics parameters
        /// </summary>
        /// <param name="erosionRate"></param>
        /// <param name="reinforcementRate"></param>
        /// <param name="decayRate"></param>
        /// <param name="confidenceThreshold"></param>
        /// <param name="prunningThreshold"></param>
        public void configureReactiveHelper(
            double erosionRate,
            double reinforcementRate,
            double decayRate,
            double confidenceThreshold,
            int prunningThreshold)
        {
            // Reactive model helper dynamics variables
            this.neuronErosionRate = erosionRate;
            this.neuronReinforcementRate = reinforcementRate;
            this.neuronDecayRate = decayRate;
            this.neuronConfidenceThreshold = confidenceThreshold;
            this.neuronPrunningThreshold = prunningThreshold;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Inserts a new neuron cluster into the ANN
        /// </summary>
        private void createNeuron()
        {
            double[][] neuron = new double[totalFields][];

            for (int field = 0; field < totalFields; field++)
            {
                neuron[field] = new double[featuresSizes[field]];
                for (int element = 0; element < featuresSizes[field]; element++)
                {
                    neuron[field][element] = 1.0;
                }
            }

            neurons.Add(neuron);
            neuronsConfidence.Add(1.0);
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Set the activity to operate 
        /// </summary>
        /// <param name="stimulus"></param>
        public void setActivity(double[][] stimulus)
        {
            for (int field = 0; field < totalFields; field++)
                Array.Copy(stimulus[field], 0, activity[field], 0, featuresSizes[field]);
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Reads the current activity
        /// </summary>
        /// <param name="field"></param>
        /// <returns></returns>
        public double[] readAcitivity(int field)
        {
            return (double[])activity[field].Clone();
        }
        //----------------------------------------------------------------------------------------------------        
        /// <summary>
        /// Reads the current activity
        /// </summary>
        private void calculateActivitySum()
        {
            for (int field = 0; field < totalFields; field++)
            {
                activitySum[field] = 0.0;
                for (int element = 0; element < featuresSizes[field]; element++)
                    activitySum[field] += activity[field][element];
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Calculates the fuzzy ART I categorization measurement 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="neuronField"></param>
        /// <param name="neuron_wAndxSum"></param>
        /// <returns></returns>
        private double ARTI(int field, double[] neuronField, double[] neuron_wAndxSum)
        {
            double xAndwSum = 0.0;
            double wSum = 0.0;
            for (int element = 0; element < featuresSizes[field]; element++)
            {
                xAndwSum += Math.Min(neuronField[element], activity[field][element]);
                wSum += neuronField[element];
            }

            neuron_wAndxSum[field] = xAndwSum;

            double t = (xAndwSum / (alphas[field] + wSum));

            if (t == Double.NaN)
                t = 0.00001;

            return t * gammas[field];
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Calculates the fuzzy ART II categorization measurement
        /// </summary>
        /// <param name="field"></param>
        /// <param name="neuronField"></param>
        /// <param name="neuron_wAndxSum"></param>
        /// <returns></returns>
        private double ARTII(int field, double[] neuronField, double[] neuron_wAndxSum)
        {
            double xAndwSum = 0.0;
            double xDotw = 0.0;
            double wLenght = 0.0;
            double xLenght = 0.0;

            for (int element = 0; element < featuresSizes[field]; element++)
            {
                xAndwSum += Math.Min(neuronField[element], activity[field][element]);
                xDotw += neuronField[element] * activity[field][element];
                wLenght += Math.Pow(neuronField[element], 2.0);
                xLenght += Math.Pow(activity[field][element], 2.0);
            }

            neuron_wAndxSum[field] = xAndwSum;

            wLenght = Math.Sqrt(wLenght);
            xLenght = Math.Sqrt(xLenght);

            double t = xDotw / (alphas[field] + (wLenght * xLenght));

            if (t == Double.NaN)
                t = 0.00001;

            return t * gammas[field];
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Calculates the proximity categorization measurement 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="neuronField"></param>
        /// <param name="neuron_wAndxSum"></param>
        /// <returns></returns>
        private double proximity(int field, double[] neuronField, double[] neuron_wAndxSum)
        {
            double xAndwSum = 0.0;
            double dist = 0.0;

            for (int element = 0; element < featuresSizes[field]; element++)
            {
                xAndwSum += Math.Min(neuronField[element], activity[field][element]);
                dist += Math.Abs(neuronField[element] - activity[field][element]);
            }

            double t = 1.0 / (alphas[field] + dist);
            neuron_wAndxSum[field] = xAndwSum;
            // neuron_wAndxSum[field] = 1.0 - (dist / (double)
            // featuresSizes[field]);

            if (t == Double.NaN)
                t = 0.000001;

            return t * gammas[field];
        }
        //----------------------------------------------------------------------------------------------------        
        /// <summary>
        /// Calculates composite operation 
        /// </summary>
        /// <param name="neuron"></param>
        /// <param name="neuron_wAndxSum"></param>
        /// <returns></returns>
        private double calculateTComposite(double[][] neuron, double[] neuron_wAndxSum)
        {
            double t = 0.0;
            for (int field = 0; field < totalFields; field++)
            {
                if (activeFields[field])
                {
                    int op = temperatureOp[field];
                    switch (op)
                    {
                        case 1:
                            t += ARTI(field, neuron[field], neuron_wAndxSum);
                            break;
                        case 2:
                            t += ARTII(field, neuron[field], neuron_wAndxSum);
                            break;
                        case 3:
                            t += proximity(field, neuron[field], neuron_wAndxSum);
                            break;
                    }
                }
            }

            return t;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Calculates the match factor between the neuron cluster weights and the received stimulus ones 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="selectedNeuronField"></param>
        /// <param name="wAndxSum"></param>
        /// <returns></returns>
        private double doMatch(int field, double[] selectedNeuronField, double wAndxSum)
        {
            double m_j = wAndxSum / activitySum[field];

            return m_j;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Verifies if a perfect miss match occurred, when the verified stimulus is 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="neuron"></param>
        /// <returns></returns>
        private bool checkPerfectMissmatch(int field, double[][] neuron)
        {
            bool pmm = true;
            for (int i = 0; i < neuron[field].Length; i++)
            {
                if (neuron[field][i] != activity[field][i])
                    pmm = false;
            }

            return pmm;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Stamps the received stimulus into the selected neuron cluster weights with the fuzzy ART I operation 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="selectedNeuron"></param>
        private void stampNeuronARTI(int field, int selectedNeuron)
        {
            double[][] learningNeuron = neurons[selectedNeuron];
            for (int element = 0; element < featuresSizes[field]; element++)
            {
                double learnedValue = (1.0 - learningRate[field]) * learningNeuron[field][element]
                        + learningRate[field] * Math.Min(learningNeuron[field][element], activity[field][element]);
                learningNeuron[field][element] = learnedValue;
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Stamps the received stimulus into the selected neuron cluster weights with the fuzzy ART II operation
        /// </summary>
        /// <param name="field"></param>
        /// <param name="selectedNeuron"></param>
        private void stampNeuronARTII(int field, int selectedNeuron)
        {
            double[][] learningNeuron = neurons[selectedNeuron];
            for (int element = 0; element < featuresSizes[field]; element++)
            {
                double learnedValue = (1.0 - learningRate[field]) * learningNeuron[field][element]
                        + learningRate[field] * activity[field][element];
                learningNeuron[field][element] = learnedValue;
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the composite learn operation 
        /// </summary>
        /// <param name="selectedNeuron"></param>
        private void learnComposite(int selectedNeuron)
        {
            if (neurons[selectedNeuron][totalFields - 1][0] == 1.0
                    && neurons[selectedNeuron][totalFields - 1][1] == 0.0 && fieldsClass[totalFields - 1] == REWARD)
                return;

            if (neurons[selectedNeuron][totalFields - 1][0] == 0.0
                    && neurons[selectedNeuron][totalFields - 1][1] == 1.0 && fieldsClass[totalFields - 1] == REWARD)
                return;

            for (int field = 0; field < totalFields; field++)
            {
                int op = learningOp[field];
                switch (op)
                {
                    case 1:
                        stampNeuronARTI(field, selectedNeuron);
                        break;
                    case 2:
                        stampNeuronARTII(field, selectedNeuron);
                        break;
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Resets the last selected neuron action. Helps the action selection when using a reactive model 
        /// </summary>
        public void resetLastNeuronAction()
        {
            double[][] learningNeuron = neurons[lastSelectedNeuron];
            for (int field = 0; field < totalFields; field++)
            {
                if (fieldsClass[field] == ACTION)
                {
                    for (int element = 0; element < featuresSizes[field]; element++)
                    {
                        double learnedValue = activity[field][element];
                        learningNeuron[field][element] = learnedValue;
                    }
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Overwrites the selected neuron cluster weights with the received stimulus 
        /// </summary>
        /// <param name="selectedNeuron"></param>
        private void overwrite(int selectedNeuron)
        {
            double[][] learningNeuron = neurons[selectedNeuron];
            for (int field = 0; field < totalFields; field++)
            {
                for (int element = 0; element < featuresSizes[field]; element++)
                {
                    double learnedValue = activity[field][element];
                    learningNeuron[field][element] = learnedValue;
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs a readout operation to the current activity using the fuzzy ART I operation 
        /// </summary>
        /// <param name="selectedNeuron"></param>
        private void ARTIReadout(int selectedNeuron)
        {
            double[][] readoutneuron = neurons[selectedNeuron];
            for (int field = 0; field < totalFields; field++)
            {
                for (int element = 0; element < featuresSizes[field]; element++)
                {
                    activity[field][element] = Math.Min(readoutneuron[field][element], activity[field][element]);
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs a readout operation to the current activity using the fuzzy ART II operation 
        /// </summary>
        /// <param name="selectedNeuron"></param>
        private void readout(int selectedNeuron)
        {
            double[][] readoutneuron = neurons[selectedNeuron];
            for (int field = 0; field < totalFields; field++)
            {
                for (int element = 0; element < featuresSizes[field]; element++)
                {
                    activity[field][element] = readoutneuron[field][element];
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Returns the total number of neurons used by the ANN 
        /// </summary>
        /// <returns></returns>
        public int getTotalAmountOfNeurons()
        {
            return neurons.Count;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the neuron prunning procedure when acting through a reactive model 
        /// </summary>
        public void neuronPrunning()
        {
            int neuronsToPrunne = 0;

            if (neurons.Count >= neuronPrunningThreshold)
            {
                for (int currentNeuron = 0; currentNeuron < neuronsConfidence.Count; currentNeuron++)
                {
                    double confidence = neuronsConfidence[currentNeuron];
                    if (confidence < neuronConfidenceThreshold)
                    {
                        neurons[currentNeuron] = null;
                        neuronsToPrunne++;
                    }
                }
            }

            while (neuronsToPrunne > 0)
            {
                for (int currentNeuron = 0; currentNeuron < neurons.Count - 1; currentNeuron++)
                {
                    if (neurons[currentNeuron] == null)
                    {
                        neurons.RemoveAt(currentNeuron);
                        neuronsConfidence.RemoveAt(currentNeuron);
                        neuronsToPrunne--;
                        break;
                    }
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the neuron cluster reinforcement operation when using a reactive model 
        /// </summary>
        public void neuronReinforcement()
        {
            double oldConfidence = neuronsConfidence[lastSelectedNeuron];
            double newConfidence = oldConfidence + neuronReinforcementRate * (1.0 - oldConfidence);
            neuronsConfidence[lastSelectedNeuron] = newConfidence;
        }

        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the neuron cluster erosion operation when using a reactive model 
        /// </summary>
        public void neuronErosion()
        {
            double oldConfidence = neuronsConfidence[lastSelectedNeuron];
            double newConfidence = oldConfidence - neuronErosionRate * oldConfidence;
            neuronsConfidence[lastSelectedNeuron] = newConfidence;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the neuron cluster decay operation when using a reactive model 
        /// </summary>
        public void neuronDecay()
        {
            for (int currentNeuron = 0; currentNeuron < neuronsConfidence.Count; currentNeuron++)
            {
                double confidence = neuronsConfidence[currentNeuron];
                double newConfidence = confidence - neuronDecayRate * confidence;
                neuronsConfidence[currentNeuron] = newConfidence;
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Set the vigilance parameters to perform or learn 
        /// </summary>
        /// <param name="learning"></param>
        /// <returns></returns>
        private double[] calculateVigilances(bool learning)
        {
            if (learning)
                return (double[])learnVigilances.Clone();
            return (double[])performVigilances.Clone();
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Return the highest value from a double array
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        private static int max(double[] array)
        {
            int result = -1;
            double max = Double.MinValue;
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] >= max)
                {
                    max = array[i];
                    result = i;
                }
            }
            return result;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Select an action if using an action mask model 
        /// </summary>
        /// <returns></returns>
        public int selectAction()
        {
            prediction(false);

            if (lastSelectedNeuron == neurons.Count - 1) // Predição de neuronio
                return -1;

            int actionField = 0;
            for (int i = 0; i < totalFields; i++)
            {
                if (fieldsClass[i] == ACTION)
                {
                    actionField = i;
                    break;
                }
            }

            int selectedAct = max(activity[actionField]);
            return selectedAct;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Check if the neuron cluster is resonating with the received stimulus 
        /// </summary>
        /// <param name="vigilances"></param>
        /// <param name="neuron"></param>
        /// <param name="neuronXandWSum"></param>
        /// <returns></returns>
        private bool isResonating(double[] vigilances, double[][] neuron, double[] neuronXandWSum)
        {
            bool resonated = true;
            for (int field = 0; field < totalFields; field++)
            {
                if (activeFields[field])
                {
                    double stateMatchFactor = doMatch(field, neuron[field], neuronXandWSum[field]);

                    if (stateMatchFactor < vigilances[field])
                    {
                        resonated = false;
                        break;
                    }
                }
            }
            return resonated;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Insert, as a copy, the perceived stimulus into the last selected neuron cluster 
        /// </summary>
        public void insert()
        {
            double[][] learningNeuron = neurons[lastSelectedNeuron];
            for (int field = 0; field < totalFields; field++)
            {
                for (int element = 0; element < featuresSizes[field]; element++)
                {
                    double learnedValue = activity[field][element];
                    learningNeuron[field][element] = learnedValue;
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the prediction operation to select a neuron cluster as a retrieved memory 
        /// </summary>
        /// <param name="learning"></param>
        public void prediction(bool learning)
        {
            calculateActivitySum();

            double[] vigilances = calculateVigilances(learning);

            double[][] wAndxSum = new double[neurons.Count][];
            for (int i = 0; i < wAndxSum.Length; i++)
                wAndxSum[i] = new double[totalFields];

            List<NeuronTemperatureTuple> neuronsTemperature = new List<NeuronTemperatureTuple>();
            for (int currentNeuron = 0; currentNeuron < neurons.Count - 1; currentNeuron++)
            {
                double t = calculateTComposite(neurons.[currentNeuron], wAndxSum[currentNeuron]);
                neuronsTemperature.Add(new NeuronTemperatureTuple(currentNeuron, t));
            }
            calculateTComposite(neurons[neurons.Count - 1], wAndxSum[neurons.Count - 1]);
            neuronsTemperature.Add(new NeuronTemperatureTuple(neurons.Count - 1, 0.0));

            neuronsTemperature.Sort((first, second) =>
            {
                if (first != null && second != null)
                    return first.t.CompareTo(second.t);

                if (first == null && second == null)
                    return 0;

                if (first != null)
                    return -1;

                return 1;
            });

            int selectedNeuron = -1;
            bool perfectMissmatch = false;
            for (int i = 0; i < neuronsTemperature.Count; i++)
            {
                int maxT = neuronsTemperature[i].neuronIndex;
                double[][] neuron = neurons[maxT];
                double[] neuronXandWSum = wAndxSum[maxT];
                selectedNeuron = maxT;

                bool resonated = isResonating(vigilances, neuron, neuronXandWSum);

                if (resonated)
                {
                    break;
                }
                else
                {
                    bool perfectError = true;
                    for (int field = 0; field < totalFields; field++)
                    {
                        if (!checkPerfectMissmatch(field, neuron) && fieldsClass[field] == STATE)
                        {
                            perfectError = false;
                            break;
                        }
                    }

                    if (perfectError)
                    {
                        perfectMissmatch = true;
                        break;
                    }
                    else
                    {
                        for (int field = 0; field < totalFields; field++)
                        {
                            if (vigilancesRaising[field])
                            {
                                double stateMatchFactor = doMatch(field, neuron[field], neuronXandWSum[field]);
                                if (stateMatchFactor > vigilances[field])
                                    vigilances[field] = Math.Min(stateMatchFactor + adaptiveVigilanceRate, 1.0);
                            }
                        }
                    }
                }
            }

            if (learning)
            {
                if (learningEnabled)
                {
                    if (perfectMissmatch)
                        overwrite(selectedNeuron);
                    else
                        learnComposite(selectedNeuron);

                    if (selectedNeuron == neurons.Count - 1)
                        createNeuron();
                }
            }
            else
            {
                if (fuzzyReadout)
                    ARTIReadout(selectedNeuron);
                else
                    readout(selectedNeuron);
            }

            lastSelectedNeuron = selectedNeuron;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Returns the last obtained prediction, stored in the activated neuron cluster 
        /// </summary>
        /// <returns></returns>
        public double[][] getLastActivatedPrediction()
        {
            return neurons[lastSelectedNeuron];
        }
        //----------------------------------------------------------------------------------------------------

        // Field type variables
        private static int STATE = 0;
        private static int ACTION = 1;
        private static int REWARD = 2;

        // Structure variables
        private int totalFields;
        private int[] featuresSizes; // ok
        private int[] temperatureOp; // ok
        private int[] learningOp; // ok
        private int[] fieldsClass; // ok

        // Net dynamics variables
        private double[] learningRate; // ok
        private double[] learnVigilances; // ok
        private double[] performVigilances; // ok
        private double[] gammas; // ok
        private double[] alphas; // ok
        private double adaptiveVigilanceRate; // ok

        // Reactive model helper dynamics variables
        private double neuronErosionRate;
        private double neuronReinforcementRate;
        private double neuronDecayRate;
        private double neuronConfidenceThreshold;
        private int neuronPrunningThreshold;

        // Q-Learning model helper dynamics variables
        private double qDiscountParameter;
        private double qLearningRate;

        // Control variables
        private bool[] activeFields; // ok
        private bool[] vigilancesRaising; // ok
        private bool fuzzyReadout; // ok
        private bool learningEnabled = true;
        private int lastSelectedNeuron = 0;

        // Operation variables
        private double[][] activity;
        private double[] activitySum;

        private List<double> neuronsConfidence;
        private List<double[][]> neurons;

    }
}

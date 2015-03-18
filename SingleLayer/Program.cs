using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SingleLayer
{
	class Program
	{
		private class Neuron
		{
			public int[] Weights { get; private set; }

			public int Bias { get; private set; }

			private static int StepFunction(int x)
			{
				if (x >= 0)
					return 1;
				else
					return 0;
			}

			public Neuron(int inputSize)
			{
				Weights = new int[inputSize];
			}

			public int Output(int[] input)
			{
				var sigma = input.Zip(Weights,(Xi,Wi) => Xi * Wi).Sum();
				return StepFunction(sigma + Bias);
			}

			public void Train(int[] input, int correctResult)
			{
				int result = Output(input);
				Bias += (correctResult - result);
				for (int i = 0; i < Weights.Length; i++)
				{
					Weights[i] += (correctResult - result) * input[i];
				}
			}
		}

		static void Main(string[] args)
		{
			var trainingForAND = new Dictionary<int[],int>
			{
				{ new[]{0 , 0}, 0 },
				{ new[]{0 , 1}, 0 },
				{ new[]{1 , 0}, 0 },
				{ new[]{1 , 1}, 1 },
			};

			var trainingForOr = new Dictionary<int[],int>
			{
				{ new[]{0 , 0}, 0 },
				{ new[]{0 , 1}, 1 },
				{ new[]{1 , 0}, 1 },
				{ new[]{1 , 1}, 1 },
			};

			var trainingForNot = new Dictionary<int[], int>
			{
				{ new[] { 0 }, 1 },
				{ new[] { 1 }, 0 }
			};

			var needsToTrain = true;

			var neuronForAnd = new Neuron(2);
			var neuronForOr = new Neuron(2);
			var neuronForNot = new Neuron(1);

			Train(trainingForAND, neuronForAnd);
			Train(trainingForOr, neuronForOr);
			Train(trainingForNot, neuronForNot);

			Console.WriteLine("--");
			foreach (var inputSet in trainingForAND)
			{
				Console.WriteLine("{0} AND {1} is {2}",inputSet.Key[0],inputSet.Key[1], neuronForAnd.Output(inputSet.Key));
			}

			Console.WriteLine("--");
			foreach (var inputSet in trainingForOr)
			{
				Console.WriteLine("{0} OR {1} is {2}", inputSet.Key[0], inputSet.Key[1], neuronForOr.Output(inputSet.Key));
			}
			Console.WriteLine("--");

			Console.WriteLine("NOT 0 is {0}",neuronForNot.Output(new[]{ 0 }));
			Console.WriteLine("NOT 1 is {0}",neuronForNot.Output(new[]{ 1 }));
		}

		private static void Train(Dictionary<int[], int> trainingSets, Neuron n, int epochCount = 1000)
		{
			var epochIndex = 0;
			var epochHasErrors = false;
			var epochCountWithoutError = 0;
			while (epochIndex < epochCount || epochHasErrors == true)
			{
				foreach (var inputSet in trainingSets)
				{
					n.Train(inputSet.Key, inputSet.Value);
					if (n.Output(inputSet.Key) != inputSet.Value)
						epochHasErrors = true;
				}

				if (epochHasErrors == false)
					epochCountWithoutError++;
				else
					epochCountWithoutError = 0;

				if (epochCountWithoutError > 5)
					break; //at least 10 full epochs without errors -> we finished training early
				epochIndex++;
			}

			Console.WriteLine("finished training after {0} epochs. Weights:{1}, Bias:{2}", epochIndex + 1,String.Join(",",n.Weights),n.Bias);
		}


	}
}

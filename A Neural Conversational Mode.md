# A Neural Conversational Mode



Conversational modeling is an important task in natural language understanding and machine intelligence. The model used here converses by predicting the next sentence given the previous sentence or sentences in a conversation.

 Neural networks can do more than just mere classification,like the task of mapping a sequence to another sequence which has direct applications in natural language understanding. In this research work, they experimented with the conversation modeling task by casting it to a task of predicting the next sequence given the previous sequence or sequences using recurrent networks. It is observed that this approach does surprisingly well on generating fluent and accurate replies to conversations. The model was tested on chat sessions from an IT helpdesk dataset of conversations, and it was found that the model can sometimes track the problem and provide a useful answer to the user. The model was also experimented with conversations obtained from a noisy dataset of movie subtitles, and it was found that the model could hold a natural conversation and sometimes perform simple forms of common sense reasoning.

 It is known that vanilla RNNs suffer from vanishing gradients, so most researchers use variants of Long Short Term Memory (LSTM) recurrent neural networks. It is also seen that recurrent neural networks are effective models for natural language. This research is based on producing answers given by a probabilistic model trained to maximize the probability of the answer given some context.

### Model

The model is based on a recurrent neural network which reads the input sequence one token at a time, and predicts the output sequence, also one token at a time. During training, the true output sequence is given to the model, so learning can be done by back propagation. During inference, given that the true output sequence is not observed, the predicted output token is simply fed as input to predict the next output. This model can be used for machine translation, question/answering, and conversations without major changes in the architecture. The input sequence can be the concatenation of what has been conversed so far (the context), and the output sequence is the reply.

### Datasets

Two datasets have been used in this experiment: a closed-domain IT helpdesk troubleshooting dataset and an open-domain movie transcript dataset. In the first set of experiments,IT Helpdesk Troubleshooting dataset is used in which costumers face computer related issues, and a specialist help them by conversing and walking through a solution.

The training set contains 30M tokens, and 3M tokens were used as validation. Some amount of clean up was performed, such as removing common names, numbers, and full URLs. The second set of experiments is performed on Open-Subtitles dataset. The dataset consists of movie conversations in XML format. They trained the model to predict the next sentence given the previous one, and did this for every sentence.

The training and validation split has 62M sentences as training examples, and the validation set has 26M sentences. The split was done in such a way that each sentence in a pair of sentences either appear together in the training set or test set but not both.

### Experiments

In the IT Helpdesk Troubleshooting dataset, they have trained a single layer LSTM with 1024 memory cells using stochastic gradient descent with gradient clipping. The model achieved a perplexity of 8, whereas an n-gram model achieved 18. In the troubleshooting session conversations, Machine is the Neural Conversational Model, and Human the human actor interacting with it. It is seen that the model solves the problem stated by the human. The problems the model was used for are VPN issues,Browser issues and Password issues.

 In Open-Subtitles dataset experiment, they have trained a two-layered LSTM using AdaGrad with gradient clipping. Each layer of the LSTM has 4096 memory cells, and we built a vocabulary consisting of the most frequent 100K words. The smoothed 5-gram model achieves a perplexity of 28.In addition to the perplexity measure, the simple recurrent model does often produce plausible answers. The model was used on a few problems - Basic Conversation,Simple Q&A, General knowledge Q&A, Philosophical Q&A, Morality,Opinions. They found it encouraging that the model can remember facts, understand contexts, perform common sense reasoning without the complexity in traditional pipelines. Perhaps most practically significant is the fact that the model can generalize to new questions. Nonetheless there are a few drawbacks. The basic model only gives simple, short, sometimes unsatisfying answers to questions, the model does not capture a consistent personality. This problem was faced during the Job and Personality conversation test. In order to fairly and objectively compare the model against CleverBot, they picked 200 questions, and asked four different humans to rate the model (NCM) versus CleverBot (CB).It was found that NCM model was preferred in 97 out of 200 questions, whereas CleverBot was picked in 60 out of 200 but they still believe there is scope for improvement. 

### Conclusion

Even though the model has obvious limitations, it is surprising to that a purely data driven approach without any rules can produce rather proper answers to many types of questions. However, the model may require substantial modifications to be able to deliver realistic conversations. Amongst the many limitations, the lack of a coherent personality makes it difficult for the system to pass the Turing test.
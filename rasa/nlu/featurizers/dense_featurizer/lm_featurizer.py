from __future__ import annotations
import numpy as np
import logging

from typing import Any, Text, List, Dict, Tuple, Type
import tensorflow as tf

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    SEQUENCE_FEATURES,
    SENTENCE_FEATURES,
    NO_LENGTH_RESTRICTION,
    NUMBER_OF_SUB_TOKENS,
    TOKENS_NAMES,
)
from rasa.shared.nlu.constants import TEXT, ACTION_TEXT
from rasa.utils import train_utils

logger = logging.getLogger(__name__)

MAX_SEQUENCE_LENGTHS = {
    "bert": 512,
    "gpt": 512,
    "gpt2": 512,
    "xlnet": NO_LENGTH_RESTRICTION,
    "distilbert": 512,
    "roberta": 512,
}


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class LanguageModelFeaturizer(DenseFeaturizer, GraphComponent):
    """A featurizer that uses transformer-based language models.

    This component loads a pre-trained language model
    from the Transformers library (https://github.com/huggingface/transformers)
    including BERT, GPT, GPT-2, xlnet, distilbert, and roberta.
    It also tokenizes and featurizes the featurizable dense attributes of
    each message.
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [Tokenizer]

    def __init__(
        self, config: Dict[Text, Any], execution_context: ExecutionContext
    ) -> None:
        """Initializes the featurizer with the model in the config."""
        super(LanguageModelFeaturizer, self).__init__(
            execution_context.node_name, config
        )
        self._load_model_metadata()
        self._load_model_instance() #yd。功能：下载LanguageModelFeaturizer组件对应的模型，例如BertTokenier分词模型和Bert词向量表示模型

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns LanguageModelFeaturizer's default config."""
        return {
            **DenseFeaturizer.get_default_config(),
            # name of the language model to load.
            "model_name": "bert",
            # Pre-Trained weights to be loaded(string)
            "model_weights": None,
            # an optional path to a specific directory to download
            # and cache the pre-trained model weights.
            "cache_dir": None,
        }

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates the configuration."""
        pass

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> LanguageModelFeaturizer:
        """Creates a LanguageModelFeaturizer.

        Loads the model specified in the config.
        """
        return cls(config, execution_context)

    @staticmethod
    def required_packages() -> List[Text]:
        """Returns the extra python dependencies required."""
        return ["transformers"]

    def _load_model_metadata(self) -> None:
        """Loads the metadata for the specified model and set them as properties.

        This includes the model name, model weights, cache directory and the
        maximum sequence length the model can handle.
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_class_dict,
            model_weights_defaults,
        )

        self.model_name = self._config["model_name"]

        if self.model_name not in model_class_dict:
            raise KeyError(
                f"'{self.model_name}' not a valid model name. Choose from "
                f"{str(list(model_class_dict.keys()))} or create"
                f"a new class inheriting from this class to support your model."
            )

        self.model_weights = self._config["model_weights"]
        self.cache_dir = self._config["cache_dir"]

        if not self.model_weights:
            logger.info(
                f"Model weights not specified. Will choose default model "
                f"weights: {model_weights_defaults[self.model_name]}"
            )
            self.model_weights = model_weights_defaults[self.model_name]

        self.max_model_sequence_length = MAX_SEQUENCE_LENGTHS[self.model_name]

    def _load_model_instance(self) -> None:
        """Tries to load the model instance.

        Model loading should be skipped in unit tests.
        See unit tests for examples.
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_class_dict,
            model_tokenizer_dict,
        )

        logger.debug(f"Loading Tokenizer and Model for {self.model_name}")

        self.tokenizer = model_tokenizer_dict[self.model_name].from_pretrained(
            self.model_weights, cache_dir=self.cache_dir
        ) #yd。加载指定的分词器（例如BertTokenizer）
        self.model = model_class_dict[self.model_name].from_pretrained(  # type: ignore[no-untyped-call] # noqa: E501
            self.model_weights, cache_dir=self.cache_dir
        )

        # Use a universal pad token since all transformer architectures do not have a
        # consistent token. Instead of pad_token_id we use unk_token_id because
        # pad_token_id is not set for all architectures. We can't add a new token as
        # well since vocabulary resizing is not yet supported for TF classes.
        # Also, this does not hurt the model predictions since we use an attention mask
        # while feeding input.
        self.pad_token_id = self.tokenizer.unk_token_id

    def _lm_tokenize(self, text: Text) -> Tuple[List[int], List[Text]]:
        """Passes the text through the tokenizer of the language model.
            #yd。lm_tokenize即language_model_tokenize，用于对text中的每个字符进行编码，得到每个字符的id
        Args:
            text: Text to be tokenized.
                  #yd。即要被分词的文本，例如'你好'
        Returns: List of token ids and token strings.
                 split_token_ids #由text中每个字符的token_id组成的list，例如[872, 1962]
                 split_token_strings #有token_id得到的字符组成的list，例如['你', '好']
        """
        split_token_ids = self.tokenizer.encode(text, add_special_tokens=False) #yd。对text进行编码，得到每个字符的id，组成split_token_ids

        split_token_strings = self.tokenizer.convert_ids_to_tokens(split_token_ids) #yd。将split_token_ids转换为字符

        return split_token_ids, split_token_strings

    def _add_lm_specific_special_tokens(
        self, token_ids: List[List[int]]
    ) -> List[List[int]]:
        """Adds the language and model-specific tokens used during training.
           #yd。根据self.model_name给每个batch的token_ids这个list首尾加上特殊token id。如果self.model_name是bert，则给token_ids这个list首尾加上CLS和SEP标识
        Args:
            token_ids: List of token ids for each example in the batch.

        Returns: Augmented list of token ids for each example in the batch.
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_special_tokens_pre_processors,
        )

        augmented_tokens = [
            model_special_tokens_pre_processors[self.model_name](example_token_ids)
            for example_token_ids in token_ids
        ]
        return augmented_tokens

    def _lm_specific_token_cleanup(
        self, split_token_ids: List[int], token_strings: List[Text]
    ) -> Tuple[List[int], List[Text]]:
        """Cleans up special chars added by tokenizers of language models.
        #yd。功能：很多语言模型会在一些词的前面或后面加上特殊字符，这个方法就是为了清除这些特殊字符
        Many language models add a special char in front/back of (some) words. We clean
        up those chars as they are not
        needed once the features are already computed.

        Args:
            split_token_ids: List of token ids received as output from the language
            model specific tokenizer.
            token_strings: List of token strings received as output from the language
            model specific tokenizer.

        Returns: Cleaned up token ids and token strings.
        """
        from rasa.nlu.utils.hugging_face.registry import model_tokens_cleaners

        return model_tokens_cleaners[self.model_name](split_token_ids, token_strings)

    def _post_process_sequence_embeddings(
        self, sequence_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes sentence and sequence level representations for relevant tokens.
        #yd。取每个样本CLS token对应的embedding作为句子embedding，得到sentence_embeddings。用去掉首尾CLS和SEP后剩余token的embedding组成post_processed_sequence_embeddings
        Args:
            sequence_embeddings: Sequence level dense features received as output from
            language model. #yd。表示当前batch中，每个样本的unpadding token的embedding。

        Returns: Sentence and sequence level representations.
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_embeddings_post_processors,
        )

        sentence_embeddings = []
        post_processed_sequence_embeddings = []

        for example_embedding in sequence_embeddings:
            (
                example_sentence_embedding,
                example_post_processed_embedding,
            ) = model_embeddings_post_processors[self.model_name](example_embedding)##yd。将CLS这个token对应的embedding当做是sentence_embedding。post_processed_embedding是将首尾的CLS和SEP的embedding移除后剩余的token的embedding

            sentence_embeddings.append(example_sentence_embedding)
            post_processed_sequence_embeddings.append(example_post_processed_embedding)

        return (
            np.array(sentence_embeddings),
            np.array(post_processed_sequence_embeddings),
        )

    def _tokenize_example(
        self, message: Message, attribute: Text
    ) -> Tuple[List[Token], List[int]]:
        """
        yd。功能：对message中每个词用BertTokenizer进行编码，得到每个字符的token_id，这些token_id组成token_ids_out；
                 tokens_out是由Token类对象组成的list，保存着每个词的各项属性
        :param message:
        :param attribute:
        :return:tokens_out 是由Token类对象组成的List； token_ids_out是当前message中每个词用BertTokenizer进行编码后得到的每个字的token_id组成的list
        """
        """Tokenizes a single message example.

        Many language models add a special char in front of (some) words and split
        words into sub-words. To ensure the entity start and end values matches the
        token values, use the tokens produced by the Tokenizer component. If
        individual tokens are split up into multiple tokens, we add this information
        to the respected token.

        Args:
            message: Single message object to be processed.
            attribute: Property of message to be processed, one of ``TEXT`` or
            ``RESPONSE``.

        Returns: List of token strings and token ids for the corresponding
                attribute of the message.
        """
        tokens_in = message.get(TOKENS_NAMES[attribute]) #yd。获取当前Message实例中的分词结果，即Token类实例组成的list
        tokens_out = []

        token_ids_out = []

        for token in tokens_in: #yd。这里的tokens_in进行遍历，token是一个Token类对象，保存着一个词的属性
            # use lm specific tokenizer to further tokenize the text
            split_token_ids, split_token_strings = self._lm_tokenize(token.text)#yd。即language_model_tokenize，用于对text中的字符进行编码，得到每个字符的id

            if not split_token_ids:
                # fix the situation that `token.text` only contains whitespace or other
                # special characters, which cause `split_token_ids` and
                # `split_token_strings` be empty, finally cause
                # `self._lm_specific_token_cleanup()` to raise an exception
                continue

            (split_token_ids, split_token_strings) = self._lm_specific_token_cleanup(
                split_token_ids, split_token_strings
            )#yd。很多语言模型会在一些词的前面或后面加上特殊字符，这个方法就是为了清除这些特殊字符

            token_ids_out += split_token_ids

            token.set(NUMBER_OF_SUB_TOKENS, len(split_token_strings))

            tokens_out.append(token)

        return tokens_out, token_ids_out

    def _get_token_ids_for_batch(
        self, batch_examples: List[Message], attribute: Text
    ) -> Tuple[List[List[Token]], List[List[int]]]:
        """Computes token ids and token strings for each example in batch.
           #yd。为当前batch中每个example计算example_token_ids和example_tokens，token_id是字符id。 example_tokens是由Token类对象组成的list。
        A token id is the id of that token in the vocabulary of the language model.

        Args:
            batch_examples: Batch of message objects for which tokens need to be
            computed.
            attribute: Property of message to be processed, one of ``TEXT`` or
            ``RESPONSE``.

        Returns: List of token strings and token ids for each example in the batch.
        """
        batch_token_ids = []
        batch_tokens = []
        for example in batch_examples: #yd。每个example都是一个Message类对象

            example_tokens, example_token_ids = self._tokenize_example(
                example, attribute
            )
            batch_tokens.append(example_tokens)
            batch_token_ids.append(example_token_ids)

        return batch_tokens, batch_token_ids

    @staticmethod
    def _compute_attention_mask(
        actual_sequence_lengths: List[int], max_input_sequence_length: int
    ) -> np.ndarray:
        """Computes a mask for padding tokens. #yd。功能：获取attention_mask，每个样本中存在token的位置赋1，不存在token的位置赋0，即padding的位置赋0.

        This mask will be used by the language model so that it does not attend to
        padding tokens.

        Args:
            actual_sequence_lengths: List of length of each example without any
            padding.
            max_input_sequence_length: Maximum length of a sequence that will be
            present in the input batch. This is
            after taking into consideration the maximum input sequence the model
            can handle. Hence it can never be
            greater than self.max_model_sequence_length in case the model
            applies length restriction.

        Returns: Computed attention mask, 0 for padding and 1 for non-padding
        tokens.
        """
        attention_mask = []

        for actual_sequence_length in actual_sequence_lengths:
            # add 1s for present tokens, fill up the remaining space up to max
            # sequence length with 0s (non-existing tokens)
            padded_sequence = [1] * min(
                actual_sequence_length, max_input_sequence_length
            ) + [0] * (
                max_input_sequence_length
                - min(actual_sequence_length, max_input_sequence_length)
            ) #yd。针对存在token的位置，赋1，不存在token的位置赋0，即padding的位置都赋0
            attention_mask.append(padded_sequence)

        attention_mask = np.array(attention_mask).astype(np.float32)
        return attention_mask

    def _extract_sequence_lengths(
        self, batch_token_ids: List[List[int]]
    ) -> Tuple[List[int], int]:
        """Extracts the sequence length for each example and maximum sequence length.
        #yd。获取当前batch中每个样本的实际长度，和当前batch的最大序列长度
        Args:
            batch_token_ids: List of token ids for each example in the batch.

        Returns:
            Tuple consisting of: the actual sequence lengths for each example,
            and the maximum input sequence length (taking into account the
            maximum sequence length that the model can handle.
        """
        # Compute max length across examples
        max_input_sequence_length = 0 #yd。记录当前batch中样本的最大长度
        actual_sequence_lengths = [] #yd。获取当前batch中每个样本的实际长度，保存在列表中

        for example_token_ids in batch_token_ids:
            sequence_length = len(example_token_ids) #yd。获取当前样本的实际长度
            actual_sequence_lengths.append(sequence_length)
            max_input_sequence_length = max(
                max_input_sequence_length, len(example_token_ids)
            )

        # Take into account the maximum sequence length the model can handle
        max_input_sequence_length = (
            max_input_sequence_length
            if self.max_model_sequence_length == NO_LENGTH_RESTRICTION
            else min(max_input_sequence_length, self.max_model_sequence_length)
        )

        return actual_sequence_lengths, max_input_sequence_length

    def _add_padding_to_batch(
        self, batch_token_ids: List[List[int]], max_sequence_length_model: int
    ) -> List[List[int]]:
        """Adds padding so that all examples in the batch are of the same length.
            #yd。功能：对当前batch的每个样本进行padding操作，padding至当前batch中最长样本的长度。
        Args:
            batch_token_ids: Batch of examples where each example is a non-padded list
            of token ids.
            max_sequence_length_model: Maximum length of any input sequence in the batch
            to be fed to the model.

        Returns:
            Padded batch with all examples of the same length.
        """
        padded_token_ids = []

        # Add padding according to max_sequence_length
        # Some models don't contain pad token, we use unknown token as padding token.
        # This doesn't affect the computation since we compute an attention mask
        # anyways.
        for example_token_ids in batch_token_ids:

            # Truncate any longer sequences so that they can be fed to the model
            if len(example_token_ids) > max_sequence_length_model:
                example_token_ids = example_token_ids[:max_sequence_length_model]

            padded_token_ids.append(
                example_token_ids
                + [self.pad_token_id]
                * (max_sequence_length_model - len(example_token_ids))
            )
        return padded_token_ids

    @staticmethod
    def _extract_nonpadded_embeddings(
        embeddings: np.ndarray, actual_sequence_lengths: List[int]
    ) -> np.ndarray:
        """Extracts embeddings for actual tokens. #yd。获取当前batch中，每个样本的unpadding token的embedding

        Use pre-computed non-padded lengths of each example to extract embeddings
        for non-padding tokens.

        Args:
            embeddings: sequence level representations for each example of the batch.#yd。通过Bert模型得到当前batch中每个样本中所有token的embedding，即batch_sequence_length，它的shape为(batch_size, cur_batch_max_seq_length, embedding_size)，例如(26,32,768)
            actual_sequence_lengths: non-padded lengths of each example of the batch. #yd。表示当前batch中每个样本的实际长度。

        Returns:
            Sequence level embeddings for only non-padding tokens of the batch.
        """
        nonpadded_sequence_embeddings = []
        for index, embedding in enumerate(embeddings):
            unmasked_embedding = embedding[: actual_sequence_lengths[index]]
            nonpadded_sequence_embeddings.append(unmasked_embedding)

        return np.array(nonpadded_sequence_embeddings)

    def _compute_batch_sequence_features(
        self, batch_attention_mask: np.ndarray, padded_token_ids: List[List[int]]
    ) -> np.ndarray:
        """Feeds the padded batch to the language model.
        #yd。功能：通过Bert模型得到当前batch中每个样本中所有token的embedding，即batch_hidden_states，它的shape为(batch_size, cur_batch_max_seq_length, hidden_size)，例如(26,32,768)
        Args:
            batch_attention_mask: Mask of 0s and 1s which indicate whether the token
            is a padding token or not. #yd。当前batch的attention_mask，是由0和1组成，0表示该位置是被padding的词，1表示该位置本身有词，不需要padding
            padded_token_ids: Batch of token ids for each example. The batch is padded
            and hence can be fed at once. #yd。表示当前batch中每个样本的实际长度。

        Returns:
            Sequence level representations from the language model.
        """
        model_outputs = self.model(
            tf.convert_to_tensor(padded_token_ids),
            attention_mask=tf.convert_to_tensor(batch_attention_mask),
        ) #yd。功能：通过模型，得到模型的输出结果

        # sequence hidden states is always the first output from all models #yd。取模型最后一层的输出作为sequence_hidden_states
        sequence_hidden_states = model_outputs[0] #yd。sequence_hidden_states的shape为(batch_size, cur_batch_max_seq_length, embedding_size)，例如(26,32,768)

        sequence_hidden_states = sequence_hidden_states.numpy()
        return sequence_hidden_states

    def _validate_sequence_lengths(
        self,
        actual_sequence_lengths: List[int],
        batch_examples: List[Message],
        attribute: Text,
        inference_mode: bool = False,
    ) -> None:
        """Validates sequence length. #yd。检查每个样本的序列长度是否小于模型能够处理的最大长度。

        Checks if sequence lengths of inputs are less than
        the max sequence length the model can handle.

        This method should throw an error during training, and log a debug
        message during inference if any of the input examples have a length
        greater than maximum sequence length allowed.

        Args:
            actual_sequence_lengths: original sequence length of all inputs
            batch_examples: all message instances in the batch
            attribute: attribute of message object to be processed
            inference_mode: whether this is during training or inference
        """
        if self.max_model_sequence_length == NO_LENGTH_RESTRICTION:
            # There is no restriction on sequence length from the model
            return

        for sequence_length, example in zip(actual_sequence_lengths, batch_examples):
            if sequence_length > self.max_model_sequence_length:
                if not inference_mode:
                    raise RuntimeError(
                        f"The sequence length of '{example.get(attribute)[:20]}...' "
                        f"is too long({sequence_length} tokens) for the "
                        f"model chosen {self.model_name} which has a maximum "
                        f"sequence length of {self.max_model_sequence_length} tokens. "
                        f"Either shorten the message or use a model which has no "
                        f"restriction on input sequence length like XLNet."
                    )
                logger.debug(
                    f"The sequence length of '{example.get(attribute)[:20]}...' "
                    f"is too long({sequence_length} tokens) for the "
                    f"model chosen {self.model_name} which has a maximum "
                    f"sequence length of {self.max_model_sequence_length} tokens. "
                    f"Downstream model predictions may be affected because of this."
                )

    def _add_extra_padding(
        self, sequence_embeddings: np.ndarray, actual_sequence_lengths: List[int]
    ) -> np.ndarray: #yd。这个函数的功能可以不用管。功能：将传入的sequence_embeddings放在一个新的np.array变量中。
        """Adds extra zero padding to match the original sequence length.

        This is only done if the input was truncated during the batch
        preparation of input for the model.
        Args:
            sequence_embeddings: Embeddings returned from the model #yd。由当前batch中每个样本去掉首尾CLS和SEP后剩余token的embedding组成
            actual_sequence_lengths: original sequence length of all inputs

        Returns:
            Modified sequence embeddings with padding if necessary
        """
        if self.max_model_sequence_length == NO_LENGTH_RESTRICTION:
            # No extra padding needed because there wouldn't have been any
            # truncation in the first place
            return sequence_embeddings

        reshaped_sequence_embeddings = []
        for index, embedding in enumerate(sequence_embeddings):
            embedding_size = embedding.shape[-1]
            if actual_sequence_lengths[index] > self.max_model_sequence_length:
                embedding = np.concatenate(
                    [
                        embedding,
                        np.zeros(
                            (
                                actual_sequence_lengths[index]
                                - self.max_model_sequence_length,
                                embedding_size,
                            ),
                            dtype=np.float32,
                        ),
                    ]
                )
            reshaped_sequence_embeddings.append(embedding)

        return np.array(reshaped_sequence_embeddings)

    def _get_model_features_for_batch(
        self,
        batch_token_ids: List[List[int]],
        batch_tokens: List[List[Token]],
        batch_examples: List[Message],
        attribute: Text,
        inference_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # yd。功能：①、得到当前batch中每个样本的CLS token对应的embedding，保存在sentence_embeddings中，其shape为(batch_size, embedding_size)。
        #         ②、得到每个样本文本分词结果中每个词的embedding（每个词的embedding是将涉及到的字的embedding取平均），将这些embedding保存在sequence_final_embeddings中
        """Computes dense features of each example in the batch.

        We first add the special tokens corresponding to each language model. Next, we
        add appropriate padding and compute a mask for that padding so that it doesn't
        affect the feature computation. The padded batch is next fed to the language
        model and token level embeddings are computed. Using the pre-computed mask,
        embeddings for non-padding tokens are extracted and subsequently sentence
        level embeddings are computed.

        Args:
            batch_token_ids: List of token ids of each example in the batch.
            batch_tokens: List of token objects for each example in the batch.
            batch_examples: List of examples in the batch.
            attribute: attribute of the Message object to be processed.
            inference_mode: Whether the call is during training or during inference.

        Returns:
            Sentence and token level dense representations.
        """
        # Let's first add tokenizer specific special tokens to all examples
        batch_token_ids_augmented = self._add_lm_specific_special_tokens(
            batch_token_ids
        )#yd。根据self.model_name给每个batch的token_ids这个list首尾加上特殊token id。如果self.model_name是bert，则给token_ids这个list首尾加上CLS和SEP标识

        # Compute sequence lengths for all examples
        (
            actual_sequence_lengths,
            max_input_sequence_length,
        ) = self._extract_sequence_lengths(batch_token_ids_augmented) #yd。获取当前batch中每个样本的实际长度，和当前batch的最大序列长度

        # Validate that all sequences can be processed based on their sequence
        # lengths and the maximum sequence length the model can handle
        self._validate_sequence_lengths(
            actual_sequence_lengths, batch_examples, attribute, inference_mode
        )#yd。检查每个样本的序列长度是否小于模型能够处理的最大长度。

        # Add padding so that whole batch can be fed to the model
        padded_token_ids = self._add_padding_to_batch(
            batch_token_ids_augmented, max_input_sequence_length
        )#yd。对当前batch的每个样本进行padding操作，padding至当前batch中最长样本的长度。

        # Compute attention mask based on actual_sequence_length
        batch_attention_mask = self._compute_attention_mask(
            actual_sequence_lengths, max_input_sequence_length
        )#yd。功能：获取batch_attention_mask，每个样本中存在token的位置赋1，不存在token的位置赋0，即padding的位置赋0.

        # Get token level features from the model
        sequence_hidden_states = self._compute_batch_sequence_features(
            batch_attention_mask, padded_token_ids
        )#yd。功能：通过Bert模型得到当前batch中每个样本的hidden_states，即batch_hidden_states，它的shape为(batch_size, cur_batch_max_seq_length, embedding_size)，例如(26,32,768)

        # Extract features for only non-padding tokens
        sequence_nonpadded_embeddings = self._extract_nonpadded_embeddings(
            sequence_hidden_states, actual_sequence_lengths
        )#yd。获取当前batch中，每个样本的unpadding token的embedding

        # Extract sentence level and post-processed features
        (
            sentence_embeddings,#yd。取每个样本CLS token对应的embedding作为句子embedding，得到sentence_embeddings。
            sequence_embeddings,#yd。用去掉首尾CLS和SEP后剩余token的embedding组成post_processed_sequence_embeddings
        ) = self._post_process_sequence_embeddings(sequence_nonpadded_embeddings)

        # Pad zeros for examples which were truncated in inference mode.
        # This is intentionally done after sentence embeddings have been
        # extracted so that they are not affected
        sequence_embeddings = self._add_extra_padding(
            sequence_embeddings, actual_sequence_lengths
        )#yd。功能：将传入的sequence_embeddings放在一个新的np.array变量中。这个方法可以不用管。

        # shape of matrix for all sequence embeddings
        batch_dim = len(sequence_embeddings) #yd。即batch_size
        seq_dim = max(e.shape[0] for e in sequence_embeddings) #yd。即当前batch中最长序列的长度（不包括首尾的CLS和SEP）
        feature_dim = sequence_embeddings[0].shape[1] #yd。即embedding_size，例如768
        shape = (batch_dim, seq_dim, feature_dim)

        # align features with tokens so that we have just one vector per token
        # (don't include sub-tokens)
        sequence_embeddings = train_utils.align_token_features(
            batch_tokens, sequence_embeddings, shape
        )#yd。功能：由字级别的embedding得到词级别的embedding，如果一个词是由多个字组成，则将这多个字的embedding取平均，得到对应词的embedding。最后返回词级别的token embedding

        # sequence_embeddings is a padded numpy array
        # remove the padding, keep just the non-zero vectors
        sequence_final_embeddings = []
        for embeddings, tokens in zip(sequence_embeddings, batch_tokens):
            sequence_final_embeddings.append(embeddings[: len(tokens)])
        sequence_final_embeddings = np.array(sequence_final_embeddings)

        return sentence_embeddings, sequence_final_embeddings

    def _get_docs_for_batch(
        self,
        batch_examples: List[Message],
        attribute: Text,
        inference_mode: bool = False,
    ) -> List[Dict[Text, Any]]:
        """
        yd。功能：得到当前batch中每个example的sequence_feature(shape为[词级别的token_count, embedding_size])
        和sentence_feature(shape为[1,768])，将这两种feature作为doc这个字典的value，返回doc组成是list
        :param batch_examples:
        :param attribute:
        :param inference_mode:
        :return:
        """
        """Computes language model docs for all examples in the batch.

        Args:
            batch_examples: Batch of message objects for which language model docs
            need to be computed.
            attribute: Property of message to be processed, one of ``TEXT`` or
            ``RESPONSE``.
            inference_mode: Whether the call is during inference or during training.


        Returns:
            List of language model docs for each message in batch.
        """
        batch_tokens, batch_token_ids = self._get_token_ids_for_batch(
            batch_examples, attribute
        ) #yd。batch_tokens当前batch的每个example所属的Token类对象组成的Table；batch_token_ids当前batch中每个example中的文本字符的token_id组成的Table

        (
            batch_sentence_features,
            batch_sequence_features,
        ) = self._get_model_features_for_batch(
            batch_token_ids, batch_tokens, batch_examples, attribute, inference_mode
        )# yd。功能：①、得到当前batch中每个样本的CLS token对应的embedding，保存在batch_sentence_features中，其shape为(batch_size, embedding_size)
        #          ②、得到每个样本文本分词结果中每个词的embedding（每个词的embedding是将涉及到的字的embedding取平均），将这些embedding保存在sequence_final_embeddings中


        # A doc consists of
        # {'sequence_features': ..., 'sentence_features': ...}
        batch_docs = []
        for index in range(len(batch_examples)):
            doc = {
                SEQUENCE_FEATURES: batch_sequence_features[index], #yd。每个batch_sequence_features[index]的shape为(词级别的token count，embedding_size)
                SENTENCE_FEATURES: np.reshape(batch_sentence_features[index], (1, -1)),
            }
            batch_docs.append(doc)

        return batch_docs

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """
        yd。功能
        :param training_data:
        :return:
        """
        """Computes tokens and dense features for each message in training data.

        Args:
            training_data: NLU training data to be tokenized and featurized
            config: NLU pipeline config consisting of all components.
        """
        batch_size = 64

        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

            non_empty_examples = list(
                filter(lambda x: x.get(attribute), training_data.training_examples)
            ) #yd。从training_data.training_examples的所有Message类对象的data字典中，选择attribute字段不为空的Message类对象，并将选择的结果保存在non_empty_examples中

            batch_start_index = 0

            while batch_start_index < len(non_empty_examples):

                batch_end_index = min(
                    batch_start_index + batch_size, len(non_empty_examples)
                )
                # Collect batch examples #yd。获取一个batch对应的examples
                batch_messages = non_empty_examples[batch_start_index:batch_end_index]

                # Construct a doc with relevant features
                # extracted(tokens, dense_features)
                #yd。功能：得到当前batch中每个example的sequence_feature和sentence_feature，将这两种feature作为doc这个字典的value，
                # 返回doc组成是list
                batch_docs = self._get_docs_for_batch(batch_messages, attribute)

                for index, ex in enumerate(batch_messages):
                    self._set_lm_features(batch_docs[index], ex, attribute)
                batch_start_index += batch_size

        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """Processes messages by computing tokens and dense features."""
        for message in messages:
            self._process_message(message)
        return messages

    def _process_message(self, message: Message) -> Message:
        """Processes a message by computing tokens and dense features."""
        # processing featurizers operates only on TEXT and ACTION_TEXT attributes,
        # because all other attributes are labels which are featurized during
        # training and their features are stored by the model itself.
        for attribute in {TEXT, ACTION_TEXT}:
            if message.get(attribute):
                self._set_lm_features(
                    self._get_docs_for_batch(
                        [message], attribute=attribute, inference_mode=True
                    )[0],
                    message,
                    attribute,
                )
        return message

    def _set_lm_features(
        self, doc: Dict[Text, Any], message: Message, attribute: Text = TEXT
    ) -> None:
        """Adds the precomputed word vectors to the messages features."""
        sequence_features = doc[SEQUENCE_FEATURES]#yd。sequence_feature(shape为[词级别的token_count, embedding_size])
        sentence_features = doc[SENTENCE_FEATURES]#yd。sentence_feature(shape为[1,embedding_size])
        #yd。将sequence_feature(shape为[词级别的token_count, embedding_size])和sentence_feature(shape为[1, embedding_size])
        #    都加入到Message类对象的self.features这个list中。
        self.add_features_to_message(
            sequence=sequence_features,
            sentence=sentence_features,
            attribute=attribute,
            message=message,
        )

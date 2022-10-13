from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Text

import dask

from rasa.engine.exceptions import GraphRunError
from rasa.engine.graph import ExecutionContext, GraphNode, GraphNodeHook, GraphSchema
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.storage import ModelStorage

logger = logging.getLogger(__name__)


class DaskGraphRunner(GraphRunner):
    """Dask implementation of a `GraphRunner`."""

    def __init__(
        self,
        graph_schema: GraphSchema,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> None:
        """Initializes a `DaskGraphRunner`.

        Args:
            graph_schema: The graph schema that will be run.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            execution_context: Information about the current graph run to be passed to
                each node.
            hooks: These are called before and after the execution of each node.
        """
        self._graph_schema = graph_schema
        self._instantiated_nodes: Dict[Text, GraphNode] = self._instantiate_nodes(
            graph_schema, model_storage, execution_context, hooks
        )
        self._execution_context: ExecutionContext = execution_context

    @classmethod
    def create(
        cls,
        graph_schema: GraphSchema,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> DaskGraphRunner:
        """Creates the runner (see parent class for full docstring)."""
        return cls(graph_schema, model_storage, execution_context, hooks)

    @staticmethod
    def _instantiate_nodes(
        graph_schema: GraphSchema,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> Dict[Text, GraphNode]:
        return {
            node_name: GraphNode.from_schema_node(
                node_name, schema_node, model_storage, execution_context, hooks
            )
            for node_name, schema_node in graph_schema.nodes.items()
        }

    def _build_dask_graph(self, schema: GraphSchema) -> Dict[Text, Any]:
        """Builds a dask graph from the instantiated graph.
        #yd。功能：根据graph_schema中nodes这个dict中的node_name和实例化GraphNode，构建一个新的dict，即run_graph。dict的key为node_name，value为实例化GraphNode
        For more information about dask graphs
        see: https://docs.dask.org/en/latest/spec.html
        """
        run_graph = {
            node_name: (
                self._instantiated_nodes[node_name],
                *schema_node.needs.values(),
            )
            for node_name, schema_node in schema.nodes.items()
        }
        return run_graph

    def run(
        self,
        inputs: Optional[Dict[Text, Any]] = None,
        targets: Optional[List[Text]] = None,
    ) -> Dict[Text, Any]:
        """
        yd。功能：以key-value对的形式返回node_name到GraphNode实例的映射
        :param inputs:
        :param targets:
        :return:
        """
        """Runs the graph (see parent class for full docstring)."""
        run_targets = targets if targets else self._graph_schema.target_names
        minimal_schema = self._graph_schema.minimal_graph_schema(run_targets)#yd。返回一个新的GraphSchema，在新的schema里面，所有的nodes都是一个target的后代。没有理解？？？

        # yd。功能：根据minimal_schema这个GraphSchema对象中nodes这个dict中的node_name和实例化GraphNode，构建一个新的dict，即run_graph。
        #    run_graph这个dict的key为node_name，value为实例化GraphNode
        run_graph = self._build_dask_graph(minimal_schema)

        if inputs:
            self._add_inputs_to_graph(inputs, run_graph) #yd。将inputs这个dict更新到run_graph这个字典中

        logger.debug(
            f"Running graph with inputs: {inputs}, targets: {targets} "
            f"and {self._execution_context}."
        )

        try:
            # yd。run_graph是一个dict，run_targets是一个list，这行代码的意思是将list中的每个元素作为key，从run_graph中获取对应的value。
            # 最后N个(key, value)合并成一个tuple中返回，dask_result为(('schema_validator', NluDataImporter), ('finetuning_validator', NluDataImporter), ('nlu_training_data_provider', <rasa.shared.nlu.training_data.training_data.TrainingData object at 0x0000011D32C9F8B0>), ('train_JiebaTokenizer0', FingerprintStatus(output_fingerprint='9a9e8e3e6eb045c39864b277962366f8', is_hit=True)), ('run_JiebaTokenizer0', FingerprintStatus(output_fingerprint='7e46c74d23a39f10183c503cbfdc9db3', is_hit=True)), ('run_LanguageModelFeaturizer1', FingerprintStatus(output_fingerprint=None, is_hit=False)), ('train_DIETClassifier2', FingerprintStatus(output_fingerprint=None, is_hit=False)))
            dask_result = dask.get(run_graph, run_targets)
            return dict(dask_result)
        except RuntimeError as e:
            raise GraphRunError("Error running runner.") from e

    @staticmethod
    def _add_inputs_to_graph(inputs: Optional[Dict[Text, Any]], graph: Any) -> None:
        if inputs is None:
            return

        for input_name, input_value in inputs.items():
            if isinstance(input_value, str) and input_value in graph.keys():
                raise GraphRunError(
                    f"Input value '{input_value}' clashes with a node name. Make sure "
                    f"that none of the input names passed to the `run` method are the "
                    f"same as node names in the graph schema."
                )
            graph[input_name] = (input_name, input_value)

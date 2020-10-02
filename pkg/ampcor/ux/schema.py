#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# externals
import graphene
# the package
import ampcor


# the abstractions
class Node(graphene.relay.Node):
    """
    Requirements of all flow nodes
    """

    class Meta:
        name = "Node"

    # all pyre components have a name
    name = graphene.String(required=True)
    # and a family
    family = graphene.String(required=True)

    @staticmethod
    def to_global_id(type_, id):
        print(f"Node.to_global_id: type:{type_}, id={id}")
        raise NotImplementedError("NYI!")

    @staticmethod
    def get_node_from_global_id(info, global_id, only_type=None):
        print(f"Node.get_node_from_gloabl_id: info:{info}, global_id:{global_id}")
        raise NotImplementedError("NYI!")


class Specification(graphene.Interface):
    """
    Product requirements
    """

    # flag indicating that i have data
    exist = graphene.Boolean(required=True)


class Producer(graphene.Interface):
    """
    Factory requirements
    """

    # factories have inputs
    inputs = graphene.List(graphene.NonNull(Specification))
    # and outputs
    outputs = graphene.List(graphene.NonNull(Specification))


class Workflow(graphene.Interface):
    """
    Flow requirements
    """

    # workflows contain a list of factories; everything else can be deduced from this
    factories = graphene.List(graphene.NonNull(Producer))

    @classmethod
    def resolve_type(cls, instance, info):
        # if this workflow is ampcor
        if instance.pyre_family() == "ampcor.workflows.ampcor":
            # let {graphene} know
            return Ampcor
        # we don't support any others, for now
        raise NotImplementedError("NYI!")


# the server version
class Version(graphene.ObjectType):
    """
    The server version
    """

    # the fields
    major = graphene.Int(required=True)
    minor = graphene.Int(required=True)
    micro = graphene.Int(required=True)
    revision = graphene.String(required=True)


# data products
class SLC(graphene.ObjectType):
    """
    The SLC data product
    """

    # meta
    class Meta:
        interfaces = Node,

    # the fields
    id = graphene.ID(required=True)
    name = graphene.String(required=True)
    family = graphene.String(required=True)
    shape = graphene.List(graphene.NonNull(graphene.Int), required=True)
    exists = graphene.Boolean(required=True)
    shape = graphene.List(graphene.NonNull(graphene.Int), required=True)
    data = graphene.String(required=True)

    # the resolvers
    def resolve_id(parent, info):
        return parent.pyre_spec

    def resolve_name(parent, info):
        return parent.pyre_name

    def resolve_family(parent, info):
        return parent.pyre_family()

    def resolve_exists(parent, info):
        return True


class Offsets(graphene.ObjectType):
    """
    The Offset data product
    """

    # meta
    class Meta:
        interfaces = Node,

    # the fields
    id = graphene.ID(required=True)
    name = graphene.String(required=True)
    family = graphene.String(required=True)
    exists = graphene.Boolean(required=True)
    data = graphene.String(required=True)

    # the resolvers
    def resolve_id(parent, info):
        return parent.pyre_spec

    def resolve_name(parent, info):
        return parent.pyre_name

    def resolve_family(parent, info):
        return parent.pyre_family()

    def resolve_exists(parent, info):
        return True


# the ampcor workflow
class Ampcor(graphene.ObjectType):
    """
    The flow configuration
    """

    class Meta:
        interfaces = Node, Producer, Workflow

    # the fields
    # for node
    id = graphene.ID(required=True)
    name = graphene.String(required=True)
    family = graphene.String(required=True)
    # for producer
    inputs = graphene.List(graphene.NonNull(Specification))
    outputs = graphene.List(graphene.NonNull(Specification))
    # for workflow
    factories = graphene.List(graphene.NonNull(Producer))
    # my inputs
    reference = graphene.Field(SLC, required=True)
    secondary = graphene.Field(SLC, required=True)

    # the resolvers
    def resolve_id(parent, info):
        return parent.pyre_spec

    def resolve_name(parent, info):
        return parent.pyre_name

    def resolve_family(parent, info):
        return parent.pyre_family()


# the query
class Query(graphene.ObjectType):
    """
    The top level query
    """

    # the fields
    node = Node.Field()
    flow = graphene.Field(Workflow, required=True)
    version = graphene.Field(Version, required=True)

    # the resolver
    def resolve_version(root, info):
        # get the version from the context
        return info.context.get("version")

    def resolve_flow(root, info):
        # get the version from the context
        return info.context.get("flow")


# build the schema
schema = graphene.Schema(
    # supported operations
    query=Query,
    # the concrete types in the schema
    types=[
        # workflows
        Ampcor,
        # data products
        SLC, Offsets,
        # administrative
        Version,
    ]
)


# end of file

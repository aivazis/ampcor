// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2021 all rights reserved


// externals
import { createContext, useContext } from 'react'
import { graphql, useLazyLoadQuery } from 'react-relay/hooks'

// create and publish the context
export const FlowContext = createContext(null)


// run the query
export const useFlowConfigQuery = () => useLazyLoadQuery(
    graphql`
        query flowQuery {
            # overall flow configuration
            flow {
                ... on Ampcor {
                    name
                    family
                    reference {
                        name
                        family
                        shape
                        exists
                    }
                    secondary {
                        name
                        family
                        shape
                        exists
                    }
                }
            }
        }
    `
)


// access to the entire configuration
export const useFlowContext = () => useContext(FlowContext)

// access to the inputs
// access to the output

// access to the SLC shapes
export const useReferenceShape = () => useFlowContext()?.reference.shape
export const useSecondaryShape = () => useFlowContext()?.secondary.shape


// end of file

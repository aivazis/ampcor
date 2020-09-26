// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
import { QueryRenderer } from 'react-relay'
// locals
import { environment } from '~/context'


// build a component that executes a query and passes the result to its {children} when
// it executes succesffully
const query = ({children, query, variables, whileLoading, onError}) => {
    // manage the query
    return (
        // execute the query
        <QueryRenderer
            query={query} variables={variables} environment={environment}
            render={({error, props, ...rest}) => {
                // if something went wrong
                if (error) {
                    // and there is an error handler
                    if (onError) {
                        // invoke it and render whatever it returns
                        return onError(error)
                    } else {
                        // otherwise, don't render anything
                        return null
                    }
                }

                // if no information was passed in, the query hasn't completed yet
                if (!props) {
                    // if the caller cares
                    if (whileLoading) {
                        // invoke the handler
                        return whileLoading()
                    } else {
                        // ptherwise, don't render anything
                        return null
                    }
                }

                // otherwise, all wen well and we got the answer; pass it on
                return children(props)
            }}
        />
    )
}


// publish
export default query


// end of file

// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
// support for image lazy loading
import lazysizes from 'lazysizes'
// use native lazy loading whenever possible
import 'lazysizes/plugins/native-loading/ls.native-loading'

// the component framework
import React, { Suspense } from 'react'
import ReactDOM from 'react-dom'
import { RelayEnvironmentProvider, useLazyLoadQuery } from 'react-relay/hooks'


// locals
import { environment } from '~/context'
// my root view
import { Layout } from './views'


// the inner component that schedules the startup query
const App = () => {
    // define the top level query
    const FlowQuery = graphql`
        query ampcorFlowQuery {
            # server version information
            version {
                ...server_version
            }
            flow {
                ... on Ampcor {
                    ...flow_meta
                }
            }
        }
    `
    // schedule it
    const data = useLazyLoadQuery(
        FlowQuery,                  // the query
        {},                         // variables
        {                           // configuration
            fetchPolicy: 'store-or-network'
        },
    )

    // render the app layout
    return (
        <Layout version={data.version} flow={data.flow} />
    )
}


// the outer component that sets up access to the {relay} environmet
const Root = () => (
    <RelayEnvironmentProvider environment={environment}>
        <Suspense fallback="loading ... ">
            <App />
        </Suspense>
    </RelayEnvironmentProvider>
)


// render
ReactDOM.unstable_createRoot(document.getElementById('ampcor')).render(<Root />)


// end of file

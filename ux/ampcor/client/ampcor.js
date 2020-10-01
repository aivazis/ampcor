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
import { RelayEnvironmentProvider } from 'react-relay/hooks'


// locals
import { environment } from '~/context'
// my root view
import { Layout } from './views'


// the outer component that sets up access to the {relay} environmet
const Root = () => (
    <RelayEnvironmentProvider environment={environment}>
        <Suspense fallback="loading ... ">
            <Layout />
        </Suspense>
    </RelayEnvironmentProvider>
)


// render
ReactDOM.unstable_createRoot(document.getElementById('ampcor')).render(<Root />)


// end of file

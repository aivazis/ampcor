// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved


// externals
// support for image lazy loading
import lazysizes from 'lazysizes'
// use native lazy loading whenever possible
import 'lazysizes/plugins/native-loading/ls.native-loading'

// the component framework
import React, { Suspense } from 'react'
import ReactDOM from 'react-dom'
import { RelayEnvironmentProvider } from 'react-relay/hooks'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'


// locals
// styles
import styles from './styles'
// context
import { environment, FlowContext, useFlowConfigQuery } from '~/context'
// views
import {
    Main,
    Flow, EXP, SLC, Gamma, Offsets, Plan,
    Loading, Stop
} from '~/views'


// the app layout
const App = () => {
    // get the app configuration
    const data = useFlowConfigQuery()

    // render
    return (
        <FlowContext.Provider value={data.flow}>
            <Routes>
                <Route path="/" element={<Main />} >
                    {/* the top level views */}
                    <Route path="flow" element={<Flow />} />
                    <Route path="exp" element={<EXP />} />
                    <Route path="slc*" element={<SLC />} />
                    <Route path="gamma" element={<Gamma />} />
                    <Route path="offsets" element={<Offsets />} />
                    <Route path="plan" element={<Plan />} />

                    {/* the closing page */}
                    <Route path="stop" element={<Stop />} />
                    {/* the page to render while waiting for data to arrive */}
                    <Route path="loading" element={<Loading />} />

                    {/* default landing spot */}
                    <Route index element={<Flow />} />
                </Route>
            </Routes>
        </FlowContext.Provider>
    )
}


// the outer component that sets up access to the {relay} environmet
const Root = () => (
    <RelayEnvironmentProvider environment={environment}>
        <Suspense fallback={<Loading />}>
            <Router>
                <App />
            </Router>
        </Suspense>
    </RelayEnvironmentProvider>
)


// render
ReactDOM.render(<Root />, document.getElementById('ampcor'))

// end of file

// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
// support for image lazy loading
import lazysizes from 'lazysizes'
// use native lazy loading whenever possible
import 'lazysizes/plugins/native-loading/ls.native-loading'

// the component framework
import React, { Suspense } from 'react'
import ReactDOM from 'react-dom'
import { RelayEnvironmentProvider } from 'react-relay/hooks'
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom'


// locals
// styles
import styles from './styles'
// context
import { environment, FlowContext, useFlowConfigQuery } from '~/context'
// views
import {
    Header, Footer,
    Flow, EXP, SLC, Gamma, Offsets, Plan, Loading, Stop
} from '~/views'


// the app layout
const App = () => {
    // get the app configuration
    const data = useFlowConfigQuery()

    // render
    return (
        <FlowContext.Provider value={data.flow}>
            <Router>
                <div style={styles.layout}>
                    <Header />
                    <Switch>
                        {/* the top level views */}
                        <Route path="/flow" component={Flow} />
                        <Route path="/exp" component={EXP} />
                        <Route path="/slc" component={SLC} />
                        <Route path="/gamma" component={Gamma} />
                        <Route path="/offsets" component={Offsets} />
                        <Route path="/plan" component={Plan} />

                        {/* the closing page */}
                        <Route path="/stop" component={Stop} />
                        {/* the page to render while waiting for data to arrive */}
                        <Route path="/loading" component={Loading} />

                        {/* default landing spot */}
                        <Route path="/" component={Flow} />
                    </Switch>
                    <Footer />
                </div>
            </Router>
        </FlowContext.Provider>
    )
}


// the outer component that sets up access to the {relay} environmet
const Root = () => (
    <RelayEnvironmentProvider environment={environment}>
        <Suspense fallback={<Loading />}>
            <App />
        </Suspense>
    </RelayEnvironmentProvider>
)


// render
ReactDOM.unstable_createRoot(document.getElementById('ampcor')).render(<Root />)


// end of file

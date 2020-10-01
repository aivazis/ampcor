// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom'
import { useLazyLoadQuery } from 'react-relay/hooks'


// locals
// styles
import styles from './styles'
import { FlowContext, useFlowConfigQuery } from '~/context'

// views
import {
    Header, Footer,
    Flow, EXP, SLC, Gamma, Offsets, Plan, Loading, Stop
} from '~/views'


// the layout
const Layout = () => {
    //
    const data = useFlowConfigQuery()

    // render
    return (
        <FlowContext.Provider value={data.flow}>
            <Router>
                <div style={styles.layout}>
                    <Header/>
                    <Switch>
                        {/* the top level views */}
                        <Route path="/flow" component={Flow}/>
                        <Route path="/exp" component={EXP}/>
                        <Route path="/slc" component={SLC}/>
                        <Route path="/gamma" component={Gamma}/>
                        <Route path="/offsets" component={Offsets}/>
                        <Route path="/plan" component={Plan}/>

                        {/* the closing page */}
                        <Route path="/stop" component={Stop}/>
                        {/* the page to render while waiting for data to arrive */}
                        <Route path="/loading" component={Loading}/>

                        {/* default landing spot */}
                        <Route path="/" component={Flow}/>
                    </Switch>
                    <Footer />
                </div>
            </Router>
        </FlowContext.Provider>
    )
}


// publish
export default Layout;


// end of file

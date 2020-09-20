// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
import ReactDom from 'react-dom'
import { Provider } from 'react-redux'


// my redux store
import store from './store'
// my root view
import { Layout } from './views'

// render
ReactDom.render((
    <Provider store={store}>
        <Layout/>
    </Provider>
), document.querySelector('#ampcor'))


// end of file

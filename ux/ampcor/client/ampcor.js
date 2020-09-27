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
import React from 'react'
import ReactDOM from 'react-dom'


// my root view
import { Layout } from './views'
// render
ReactDOM.unstable_createRoot(document.getElementById('ampcor')).render(<Layout />)


// end of file

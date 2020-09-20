// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import { createStore, combineReducers } from 'redux'


// locals
import navigation from './navigation'


// combine my reducers
const reducer = combineReducers({
    navigation,
})

// my store factory; it may look weird, but this will let me add middleware without disurbing
// the surrounding code...
const assemble = () => createStore(reducer)

// make my store
const store = assemble()

// and publish it
export default store


// end of file

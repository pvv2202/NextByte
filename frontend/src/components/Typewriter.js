import React from 'react'
import { TypeAnimation } from 'react-type-animation'

function Typewriter({sequence, speed}) {

  return (
    <TypeAnimation
      sequence={sequence}
      wrapper="div"
      speed={speed}
      className="text-4xl font-mono text-sky-950"
      />
  )
}

export default Typewriter
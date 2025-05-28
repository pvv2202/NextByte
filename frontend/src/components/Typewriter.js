import React from 'react'
import { TypeAnimation } from 'react-type-animation'

function Typewriter({sequence, speed, repeat=0}) {

  return (
    <TypeAnimation
      sequence={sequence}
      wrapper="div"
      speed={speed}
      className="text-4xl font-mono"
      repeat={repeat}
      cursor={false}
      />
  )
}

export default Typewriter
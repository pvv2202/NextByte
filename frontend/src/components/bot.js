import React from 'react'
import styled, {keyframes} from 'styled-components';
import {motion, AnimatePresence} from 'framer-motion'

const pulse = keyframes`
  0% {transform: scale(1.0)}
  50% {transform: scale(1.05)}
  100% {transform: scale(1.0)}
`;
export const PulseDiv = styled.div`
  animation: ${pulse} 5s ease-in-out infinite;
`;


function NextByteBot() {
  return (
    <AnimatePresence>
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ duration: 0.3}}
        exit={{ scale: 0}}
      >
        <PulseDiv>
              <img className=' max-w-60 rounded-full border-8 border-sky-200 bg-white p-8 shadow-md ' src={'../Closing-eyes.gif'} alt="nextbyte" />
        </PulseDiv>
      </motion.div>
    </AnimatePresence>
    
   
  )
}

export default NextByteBot
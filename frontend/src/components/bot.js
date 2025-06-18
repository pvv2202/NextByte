import React from 'react'
import styled, {keyframes} from 'styled-components';

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
    <PulseDiv>
      <img className=' max-w-60 rounded-full border-8 border-sky-200 bg-white p-8 shadow-md ' src={'../Closing-eyes.gif'} alt="nextbyte" />
    </PulseDiv>
  )
}

export default NextByteBot
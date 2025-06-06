import React from 'react'
import { useNavigate } from 'react-router'
import { useState } from 'react'
import { useForm } from 'react-hook-form'
import SignupForm from '../components/signupForm'
import { api_request } from '../api_request'

function SignUp() {
  const navigate = useNavigate()
  const [authenticating, setAuthenticating] = useState(false)
  const {register, handleSubmit, reset} = useForm() 
  const [userError, setUserError] = useState('')
  
  const onSubmit = async (data) => {
      setUserError('')
      setAuthenticating(true)
      try{
          const response = await api_request(
              'login',
              'POST',
              {'Content-Type': 'application/json'},
              {username: data['username'], password: data['password']}
          )
          // TODO: SET UP USER SESSION?
          navigate('/login')
          
      } catch(err){
          setUserError(err.message)
          
      }
      
      setAuthenticating(false)
      
  }
  return (
    <div className='w-full h-screen flex justify-center items-center'>
        <SignupForm
            navigate={navigate}
            authenticating={authenticating}
            register={register} handleSubmit={handleSubmit} reset={reset}
            userError={userError}
            onSubmit={onSubmit}  
        />
    </div>
  )
}

export default SignUp
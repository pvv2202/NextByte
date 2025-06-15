import React from 'react'
import { useNavigate } from 'react-router'
import { useState } from 'react'
import { useForm, Controller } from 'react-hook-form'
import SignupForm from '../components/auth/signupForm'
import { api_request } from '../api_request'

function SignUp() {
  const navigate = useNavigate()
  const [authenticating, setAuthenticating] = useState(false)
  const {register, handleSubmit, reset, control, watch} = useForm() 
  const [userError, setUserError] = useState('')
  
  const onSubmit = async (data) => {
      console.log(data)
      setUserError('')
      setAuthenticating(true)
      try{
          const response = await api_request(
              'signup',
              'POST',
              {'Content-Type': 'application/json'},
              {username: data['username'], 
                password: data['password'],
                age: data['age'],
                email: data['email'],
                country: data['country'],
                state: data['state'],
                city: data['city']

            }
          )
          // TODO: SET UP USER SESSION?
          navigate('/login')
          
      } catch(err){
          setUserError(err.message)
          
      }
      
      setAuthenticating(false)
      
  }
  return (
    <>
    <div className='w-full h-screen flex justify-center items-center gap-2'>
        <SignupForm
            navigate={navigate}
            authenticating={authenticating}
            register={register} handleSubmit={handleSubmit} reset={reset}
            userError={userError}
            onSubmit={onSubmit}  
            Controller={Controller}
            control={control}
            watch={watch}
        />
        
    </div>
    
    </>
  )
}

export default SignUp
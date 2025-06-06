import React from 'react'
import { ClipLoader } from 'react-spinners'
import { PulseDiv } from './bot'
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { api_request } from '../api_request';
import { Navigate, useNavigate } from 'react-router';


function SignupForm({handleSubmit, onSubmit, register, authenticating, userError}) {
    const navigate = useNavigate('')
    return (
        <form className="flex flex-col gap-4 bg-white p-6 rounded-xl" onSubmit={handleSubmit(onSubmit)}>
            <h2 className='flex gap-1 items-center justify-between text-gray-800 text-xl mt-4'>
                <p>Create Your Account</p>
                <PulseDiv>
                    <img className=' max-w-24 object-contain' src={'Closing-eyes.gif'} alt="nextbyte" />
                </PulseDiv>
            </h2>
            {{userError} ? <p className='text-red-600 ml-2'>{userError}</p> : null}
            <input type='text' {...register("username", {required: true,
                maxLength: 20})} 
                placeholder='Username'
                className='p-2 rounded'
            />
            <input type="text" {...register("password", {required: true, 
                maxLength:20})}
                placeholder='Password'
                className='p-2 rounded' 
            />

            




            

            <div className='flex justify-end'>
                <button type='submit' className='w-20 rounded curser:hand hover:scale-105 text-white  ml-3 p-2 bg-gray-800'>
                    {authenticating ? <ClipLoader color={'white'} speedMultiplier={0.5} size={20} /> : <p>Sign up</p>}
                </button>
            </div>
           <button onClick={() => navigate('/login')}>Already have an account?</button>
        </form>
    )
}

export default SignupForm